import os, subprocess, sys, shutil, stat, logging, shlex, getpass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Optional
# see no evil
from .workflowdata import workflowdata, workflowtests
pjoin = os.path.join

# Global configuration constants
docker = "docker" # the docker executable
srcdir = "/src" # mount location in container of source directory
blddir = "/bld" # mount location in container of build directory

# Path setup for convenience
thisdir = os.path.dirname(__file__)
projsrcdir = f"{thisdir}/../.." # XXX
runconfigdir = f"{thisdir}/runconfigs"

art_base = "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/data"
container_testdir = f"/tmp/test"

soilm_conda_env = 'SoilMoisture'

# Query docker info
docker_info = subprocess.check_output("docker info".split()).decode("utf-8")
docker_runtimes = " ".join(line for line in docker_info.split("\n") if "Runtimes:" in line)

def run_with_logging(dockercall, cmd, logger, printlog=True):
    """
    Run command as a subprocess and log the standard streams (stdout & stderr) to
    the specified logger.

    Parameters
    -------------
    dockercall : str
        Docker call and parameters to run commands with
    cmd : list
        List of command(s) to run in Docker
    logger : logger
        Python logger to log output, could be to standard out or a file
    printlog : boolean, optional
        Print log to console
    """
    logger.propagate = printlog
    # remove extra whitespace
    normalize = lambda s: subprocess.list2cmdline(shlex.split(s))
    # format command string for logging
    cmdstr = normalize(dockercall) + ' "\n' + ''.join(f'{"":<6}{normalize(c)}\n' for c in cmd) + '"'
    # save command to log
    logger.info("++ " + cmdstr + "\n")
    pipe = subprocess.Popen(shlex.split(cmdstr), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Maximum number of seconds to wait for "docker run" to finish after
    # its child process exits.  The observed times have been < 1 ms.
    # Use a relatively large number to flag a possible problem with Docker.
    timeout = 10

    with pipe.stdout:
        for line in iter(pipe.stdout.readline, b''): # b'\n'-separated lines
            decoded = line.decode("utf-8")
            # remove newline character so the log does not contain extra blank lines
            if str.endswith(decoded, '\n'):
                decoded = decoded[:-1]
            logger.info(decoded)
    ret = pipe.poll()
    if ret is None:
        ret = pipe.wait(timeout=timeout)
        # ret will be None if exception TimeoutExpired was raised and caught.
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmdstr)


def push_to_registry(
    image: str,
    server: str,
    username: str,
    password: str,
    tag: Optional[str] = None,
) -> None:
    """
    Push a docker image to a remote registry.

    Parameters
    ----------
    image : str
        The name[:tag] or ID of the existing image to push.
    server : str
        The server URL, typically in '<hostname>:<port>' format.
    username, password : str
        Credentials used to access the remote registry.
    tag : str or None, optional
        The name[:tag] to give the image in the remote registry. If None (the default),
        the value of `image` is used as the remote tag.
    """
    # Login to the docker registry.
    args = [docker, "login", f"--username={username}", "--password-stdin", server]
    subprocess.run(args, input=password, check=True, text=True)

    if tag is None:
        tag = image

    # Tag the image with the registry hostname, port, and remote tag.
    args = [docker, "tag", image, f"{server}/{tag}"]
    subprocess.run(args, check=True)

    # Push to the remote registry.
    args = [docker, "image", "push", f"{server}/{tag}"]
    subprocess.run(args, check=True)


# A set of docker images suitable for building and running isce3
class ImageSet:
    # images in the set, in the order they should be built
    imgs = [
        "runtime",
        "dev",
    ]

    cpack_generator = "RPM"

    def imgname(self, repomod="", tagmod=""):
        """
        Return unique Docker image name per Jenkins run, uses Jenkins environment variables
        JOB_NAME and EXECUTOR_NUMBER

        Parameters
        ----------
        repomod : str, optional
            Modifier to nominal Docker repository name
        tagmod : str, optional
            Modifier to nominal Docker tag name

        """
        if tagmod != "":
            tagmod = "-" + tagmod

        if self.imgtag:
            return f"nisar-adt/isce3{repomod}:{self.imgtag}{tagmod}"
        else:
            try:
                gitmod = '-' + subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode('ascii').strip()
            except:
                gitmod = ""
            return f"nisar-adt/isce3{repomod}:{self.name}{tagmod}" \
                   + f"-{getpass.getuser()}{gitmod}"

    def docker_run(self, img, cmd):
        runcmd = f"{docker} run {self.run_args} --rm -i {self.tty} " \
                 + f"{self.imgname(repomod='dev', tagmod=img)} bash -ci"
        subprocess.check_call(runcmd.split() + [cmd])

    def docker_run_dev(self, cmd):
        """
        Shortcut since most commands just run in the devel image
        """
        self.docker_run("dev", cmd)

    def __init__(self, name, *, projblddir, printlog=False, imgtag=None):
        """
        A set of docker images for building and testing isce3/nisar distributables.

        Parameters
        ----------
        name : str
            Name of the image set (e.g. "centos7", "alpine"). Used for tagging images
        projblddir : str
            Path to the binary directory on the host where build artifacts are written
        printlog : boolean, optional
            Print workflow test and qa logs to console in real-time
        """
        self.name = name
        self.projblddir = projblddir
        self.imgtag = imgtag
        self.datadir = projblddir + "/workflow_testdata_tmp/data"
        self.testdir = projblddir + "/workflow_testdata_tmp/test"
        self.build_args = f'''
            --network=host
        ''' + " ".join(f"--build-arg {x}_img={self.imgname(repomod='dev', tagmod=x)}" for x in self.imgs)

        self.cmake_defs = {
            "WITH_CUDA": "YES",
            "ISCE3_FETCH_DEPS": "NO",
            "CPACK_PACKAGE_FILE_NAME": "isce3",
        }
        self.cmake_extra_args = []

        self.run_args = f'''
            --network=host
            -v {projsrcdir}:{srcdir}:ro
            -v {projblddir}:{blddir}:rw
            -w {blddir}
            -u {os.getuid()}:{os.getgid()}
            -e DESTDIR={blddir}/install
        '''

        # Allocate a pseudo-tty if available
        self.tty = "--tty" if sys.stdin.isatty() else ""

        # Use nvidia runtime if available
        if "nvidia" in docker_runtimes:
            self.run_args += " --runtime=nvidia"

        logging.basicConfig(format='', level=logging.INFO)
        self.printlog = printlog

    def cmake_args(self):
        return [f"-D{key}={value}" for key, value in self.cmake_defs.items()] \
                + self.cmake_extra_args

    def setup(self):
        """
        Build development docker images, containing build/runtime prereqs
        (should not change often, so these will usually be cached)
        """
        for img in self.imgs:
            cmd = f"{docker} build {self.build_args} {thisdir}/{self.name}/{img}" \
                  + f" -t {self.imgname(repomod='dev', tagmod=img)}"
            subprocess.check_call(cmd.split())

    def configure(self):
        """
        Run cmake configure/generate steps
        """
        os.makedirs(self.projblddir, exist_ok=True)
        self.docker_run_dev(f'''
            . /opt/conda/etc/profile.d/conda.sh
            conda activate base
            PYPKGABS=$(python3 -c \"import sysconfig, os.path; print(sysconfig.get_paths()['purelib'])\")
            cmake {srcdir} -GNinja \
                    -DISCE_PACKAGESDIR=$PYPKGABS \
                    -DPYRE_DEST_PACKAGES=$PYPKGABS \
                    {" ".join(self.cmake_args())}
        ''')

    def build(self):
        """
        Build isce3
        """
        self.docker_run_dev(f"cmake --build {blddir} --target install")

    def test(self):
        """
        Run unit tests
        """
        self.docker_run_dev(f"""
            ctest -T Test --no-compress-output --output-on-failure || true
            """
        )

    def dropin(self):
        """
        Drop into an interactive shell, helpful for debugging
        """
        self.docker_run_dev(f"bash")

    def makepkg(self):
        """
        Create package for redistribution
        """
        self.docker_run_dev(f"cpack -G {self.cpack_generator} \
                -DCPACK_PACKAGE_RELOCATABLE=NO")

    def makedistrib(self):
        """
        Install package to redistributable isce3 docker image
        """
        subprocess.check_call(f"cp {self.projblddir}/isce3.rpm \
                {thisdir}/{self.name}/distrib/isce3.rpm".split())

        # Squashing requires experimental features
        squash_arg = "--squash" if "Experimental: true" in docker_info else ""

        cmd = f"{docker} build {self.build_args} {squash_arg} \
                    {thisdir}/{self.name}/distrib -t {self.imgname(tagmod='distrib')}"
        subprocess.check_call(cmd.split())


    def makedistrib_nisar(self):
        """
        Install package to redistributable isce3 docker image with nisar qa,
        noise estimator caltool, and Soil Moisture applications
        """
        # Get current UTC date & time in ISO 8601 format.
        creation_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")

        # Get current ISCE3 version.
        version_file = Path(projsrcdir) / "VERSION.txt"
        isce3_version = version_file.read_text().strip()

        # Get git commit hash (if available).
        args = ["git", "rev-parse", "HEAD"]
        try:
            git_commit = subprocess.check_output(args, text=True).strip()
        except subprocess.CalledProcessError:
            git_commit = ""

        build_args = {
            "distrib_img": self.imgname(tagmod='distrib'),
            "GIT_OAUTH_TOKEN": os.environ.get('GIT_OAUTH_TOKEN').strip(),
            "CREATION_DATETIME": creation_datetime,
            "ISCE3_VERSION": isce3_version,
            "GIT_COMMIT": git_commit,
        }
        build_args = " ".join(f"--build-arg {k}={v}" for (k, v) in build_args.items())

        cmd = f"{docker} build {build_args} \
                {thisdir}/{self.name}/distrib_nisar -t {self.imgname()}"
        subprocess.check_call(cmd.split())


    def fetchdata(self):
        """
        Fetch workflow testing data from Artifactory
        TODO use a properly cached download e.g. via DVC
        """


        # Download files, preserving relative directory hierarchy
        for testname, fetchfiles in workflowdata.items():
            wfdatadir = pjoin(self.datadir, testname)
            os.makedirs(wfdatadir, exist_ok=True)
            for fname in fetchfiles:
                url = f"{art_base}/{testname}/{fname}"
                print("Fetching file:", url)
                subprocess.check_call(f"curl -f --create-dirs -o {fname} -O {url} ".split(),
                                      cwd = wfdatadir)

    def fetchmindata(self):
        """
        Fetch minimum set of workflow test data from Artifactory so mintests can be exercised
        TODO use a properly cached download e.g. via DVC
        """

        mindata = [
            "L0B_RRSD_ALPSRP264757150_Amazon",
            "L0B_RRSD_REE1",
            "L0B_RRSD_REE_BF_NET",
            "L0B_RRSD_REE_CHANNEL4_EXTSCENE_PASS1",
            "L1_RSLC_REE_PTA",
            "L1_RSLC_UAVSAR_SanAnd_05024_18038_006_180730_L090_CX_129_05",
            "L1_RSLC_UAVSAR_NISARP_32039_19049_005_190717_L090_CX_129_03",
            "L1_RSLC_UAVSAR_NISARP_32039_19052_004_190726_L090_CX_129_02",
        ]

        # Download files, preserving relative directory hierarchy
        for dataset in mindata:
            wfdatadir = pjoin(self.datadir, dataset)
            os.makedirs(wfdatadir, exist_ok=True)
            for fname in workflowdata[dataset]:
                url = f"{art_base}/{dataset}/{fname}"
                print("Fetching file:", url)
                subprocess.check_call(f"curl -f --create-dirs -o {fname} -O {url} ".split(),
                                      cwd = wfdatadir)

    def distribrun(self, testdir, cmd, logfile=None, dataname=None, nisarimg=False, loghdlrname=None):
        """
        Run a command in the distributable image

        Parameters
        -------------
        testdir : str
            Test directory to run Docker command in
        cmd : list
            List of command(s) to run inside Docker
        logfile : str, optional
            File name (relative to testdir) of log file for saving standard out and standard error
        dataname : str or list, optional
            Test input data as str or list (e.g. "L0B_RRSD_REE1", ["L0B_RRSD_REE1", "L0B_RRSD_REE2")
        nisarimg : boolean, optional
            Use NISAR distributable image
        loghdlername : str, optional
            Name for logger file handler
        """
        # save stdout and stderr to logfile if specified
        if loghdlrname is None:
            loghdlrname = f'wftest.{os.path.basename(testdir)}'
        logger = logging.getLogger(name=loghdlrname)
        if logfile is None:
            hdlr = logging.StreamHandler(sys.stdout)
        else:
            hdlr = logging.FileHandler(pjoin(testdir, logfile), mode='w')
        logger.addHandler(hdlr)

        if nisarimg:
            img = self.imgname()
        else:
            img = self.imgname(tagmod="distrib")

        datamount = ""
        if dataname is not None:
            if type(dataname) is not list:
                dataname = [dataname]
            for data in dataname:
                datadir = os.path.abspath(pjoin(self.datadir, data))
                datamount += f"-v {datadir}:{container_testdir}/input_{data}:ro "

        dockercall = f"{docker} run \
            -v {testdir}:{container_testdir} {datamount} \
            -w {container_testdir} \
            -u {os.getuid()}:{os.getgid()} \
            --rm -i {self.tty} {img} sh -ci"
        run_with_logging(dockercall, cmd, logger, printlog=self.printlog)

    def workflowtest(self, wfname, testname, dataname, pyname, suf="", description="", arg=""):
        """
        Run the specified workflow test using either the distrib or the nisar image.

        Parameters
        -------------
        wfname : str
            Workflow name (e.g. "rslc")
        testname : str
            Workflow test name (e.g. "RSLC_REE1")
        dataname : str or iterable of str or None
            Test input dataset(s) to be mounted (e.g. "L0B_RRSD_REE1", ["L0B_RRSD_REE1", "L0B_RRSD_REE2"]).
            If None, no input datasets are used.
        pyname : str
            Name of the isce3 module to execute (e.g. "nisar.workflows.focus") or,
            for Soil Moisture (SM) testing, the name of the SAS executable to run
            (e.g. "NISAR_SM_SAS")
        suf: str
            Suffix in runconfig and output directory name to differentiate between
            reference and secondary data in end-to-end tests
        description: str
            Extra test description to print out to differentiate between
            reference and secondary data in end-to-end tests
        arg : str, optional
            Additional command line argument(s) to pass to the workflow
        """
        print(f"\nRunning workflow test {testname}{description}\n")
        testdir = os.path.abspath(pjoin(self.testdir, testname))
        # create input directories before docker volume mount to avoid root ownership
        # of these directories
        if dataname is not None:
            if type(dataname) is not list:
                dataname = [dataname]
            for data in dataname:
                os.makedirs(pjoin(testdir, f"input_{data}"), exist_ok=True)
        # create output directories
        os.makedirs(pjoin(testdir, f"output_{wfname}{suf}"), exist_ok=True)
        os.makedirs(pjoin(testdir, f"scratch_{wfname}{suf}"), exist_ok=True)

        # check whether we're testing one of the D&C SAS workflows (Doppler, EL Edge, or
        # EL Null products).
        is_dnc_test = (
            testname.startswith("doppler")
            or testname.startswith("el_edge")
            or testname.startswith("el_null")
        )

        # check whether we're testing one of the CalTools workflows (Noise Estimation or
        # Point Target Analysis).
        is_caltools_test = testname.startswith("noisest") or testname.startswith("pta")

        # copy test runconfig to test directory (for end-to-end testing, we need to
        # distinguish between the runconfig files for each individual workflow)
        if testname.startswith("end2end"):
            inputrunconfig = f"{testname}_{wfname}{suf}.yaml"
            shutil.copyfile(pjoin(runconfigdir, inputrunconfig),
                            pjoin(testdir, f"runconfig_{wfname}{suf}.yaml"))
        elif testname.startswith("soilm"):
            # For R3.2, the Soil Moisture SAS uses one plaintext configuration file.
            inputrunconfig = f"{testname}{suf}.txt"
            shutil.copyfile(pjoin(runconfigdir, inputrunconfig),
                            pjoin(testdir, f"runconfig_{wfname}{suf}.txt"))
        elif is_dnc_test or is_caltools_test:
            inputrunconfig = f"{testname}{suf}.txt"
            shutil.copyfile(pjoin(runconfigdir, inputrunconfig),
                            pjoin(testdir, f"runconfig_{wfname}{suf}.txt"))
        else:
            inputrunconfig = f"{testname}{suf}.yaml"
            shutil.copyfile(pjoin(runconfigdir, inputrunconfig),
                            pjoin(testdir, f"runconfig_{wfname}{suf}.yaml"))
        log = pjoin(testdir, f"output_{wfname}{suf}", "stdouterr.log")

        if testname.startswith("soilm"):
            executable = pyname
            # Execute the SoilMoisture SAS inside the Conda environment used for its build
            cmd = [f"time conda run -n {soilm_conda_env} {executable} runconfig_{wfname}{suf}.txt"]
        elif is_dnc_test or is_caltools_test:
            cmd = [f"time python3 -m {pyname} {arg} @runconfig_{wfname}{suf}.txt"]
        else:
            cmd = [f"time python3 -m {pyname} {arg} runconfig_{wfname}{suf}.yaml"]

        try:
            if not testname.startswith("soilm"):
                self.distribrun(testdir, cmd, logfile=log, dataname=dataname,
                                loghdlrname=f'wftest.{os.path.basename(testdir)}')
            else:
                # Currently, the SM executables are in the nisar image.
                self.distribrun(testdir, cmd, logfile=log, dataname=dataname, nisarimg=True,
                                loghdlrname=f"wftest.{os.path.basename(testdir)}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Workflow test {testname} failed") from e

    def doppler_test(self, tests=None):
        """Test Doppler centroid product SAS."""
        if tests is None:
            tests = workflowtests["doppler"].items()
        for testname, dataname in tests:
            self.workflowtest(
                "doppler",
                testname,
                dataname,
                "nisar.workflows.gen_doppler_range_product",
            )

    def el_edge_test(self, tests=None):
        """Test EL rising edge pointing product SAS."""
        if tests is None:
            tests = workflowtests["el_edge"].items()
        for testname, dataname in tests:
            self.workflowtest(
                "el_edge",
                testname,
                dataname,
                "nisar.workflows.gen_el_rising_edge_product",
            )

    def el_null_test(self, tests=None):
        """Test EL null range product SAS."""
        if tests is None:
            tests = workflowtests["el_null"].items()
        for testname, dataname in tests:
            self.workflowtest(
                "el_null",
                testname,
                dataname,
                "nisar.workflows.gen_el_null_range_product",
            )

    def rslctest(self, tests=None):
        if tests is None:
            tests = workflowtests['rslc'].items()
        for testname, dataname in tests:
            self.workflowtest("rslc", testname, dataname, "nisar.workflows.focus")

    def gslctest(self, tests=None):
        if tests is None:
            tests = workflowtests['gslc'].items()
        for testname, dataname in tests:
            self.workflowtest("gslc", testname, dataname, "nisar.workflows.gslc")

    def gcovtest(self, tests=None):
        if tests is None:
            tests = workflowtests['gcov'].items()
        for testname, dataname in tests:
            self.workflowtest("gcov", testname, dataname, "nisar.workflows.gcov")

    def insartest(self, tests=None):
        if tests is None:
            tests = workflowtests['insar'].items()
        for testname, dataname in tests:
            self.workflowtest("insar", testname, dataname, "nisar.workflows.insar", arg="--restart")

    def end2endtest(self, tests=None):
        """
        Run all workflows for one pair of L0B input data, including RSLC, GSLC, GCOV, RIFG, RUNW, GUNW.
        The GSLC, GCOV, and InSAR products are generated from outputs of the RSLC workflow.
        """
        if tests is None:
            tests = workflowtests['end2end'].items()
        for testname, dataname in tests:
            # copy runconfigs and create output direcotories
            testdir = os.path.abspath(pjoin(self.testdir, testname))
            for wfname in ['rslc', 'gslc', 'gcov', 'insar']:
                if wfname == 'rslc':
                    pyname = 'nisar.workflows.focus'
                else:
                    pyname = f'nisar.workflows.{wfname}'

                if wfname == 'insar':
                    self.workflowtest(wfname, testname, dataname, pyname, arg="--restart",
                                      description=" InSAR product")
                else:
                    self.workflowtest(wfname, testname, dataname, pyname, suf="_ref",
                                      description=f" {wfname.upper()} reference product")
                    self.workflowtest(wfname, testname, dataname, pyname, suf="_sec",
                                      description=f" {wfname.upper()} secondary product")


    def noisesttest(self, tests=None):
        """Test Noise Estimation tool."""
        if tests is None:
            tests = workflowtests['noisest'].items()
        for testname, dataname in tests:
            self.workflowtest(
                "noisest",
                testname,
                dataname,
                "nisar.workflows.noise_estimator",
            )

    def ptatest(self, tests=None):
        """Test Point Target Analysis tool."""
        if tests is None:
            tests = workflowtests['pta'].items()
        for testname, dataname in tests:
            self.workflowtest(
                "pta",
                testname,
                dataname,
                "nisar.workflows.point_target_analysis",
            )

    def soilmtest(self, tests=None):
        if tests is None:
            tests = workflowtests['soilm'].items()
        for testname, dataname in tests:
            # For R3.2, invoke the Soil Moisture SAS by executing a single
            # Fortran binary executable.  This executable runs the three
            # individual Soil Moisture algorithms in series to produce the
            # final soil moisture product.
            executable = 'NISAR_SM_SAS'
            self.workflowtest("soilm", testname, dataname, f"{executable}")

    def mintests(self):
        """
        Only run first test from each workflow
        """
        self.rslctest(tests=list(workflowtests['rslc'].items())[:1])
        self.doppler_test(tests=list(workflowtests['doppler'].items())[:1])
        self.el_edge_test(tests=list(workflowtests['el_edge'].items())[:1])
        self.el_null_test(tests=list(workflowtests['el_null'].items())[:1])
        self.gslctest(tests=list(workflowtests['gslc'].items())[:1])
        self.gcovtest(tests=list(workflowtests['gcov'].items())[:1])
        self.insartest(tests=list(workflowtests['insar'].items())[:1])
        self.noisesttest(tests=list(workflowtests['noisest'].items())[:1])
        self.ptatest(tests=list(workflowtests['pta'].items())[:1])

    def workflowqa(self, wfname, testname, dataname=None, suf="", description=""):
        """
        Run QA for the specified workflow using the NISAR distrib image.

        Parameters
        -------------
        wfname : str
            Workflow name (e.g. "rslc")
        testname: str
            Workflow test name (e.g. "RSLC_REE1")
        dataname : str or iterable of str or None
            Test input dataset(s) to be mounted (e.g. "L0B_RRSD_REE1", ["L0B_RRSD_REE1", "L0B_RRSD_REE2"]).
            If None, no input datasets are used.
        suf: str
            Suffix in runconfig and output directory name to differentiate between
            reference and secondary data in end-to-end tests
        description: str
            Extra test description to print out to differentiate between
            reference and secondary data in end-to-end tests
        """
        print(f"\nRunning workflow QA on test {testname}{description}\n")
        testdir = os.path.abspath(pjoin(self.testdir, testname))
        os.makedirs(pjoin(testdir, f"qa_{wfname}{suf}"), exist_ok=True)
        log = pjoin(testdir, f"qa_{wfname}{suf}", "stdouterr.log")
        # run qa command
        # XXX The QA software installs XML files which are subsequently required at
        # runtime by the QA scripts. But figuring out the installed location of these
        # files is complicated and depends on the platform & installation options. This
        # is further complicated by the fact that the QA code expects these paths to be
        # provided relative to the installed scripts location. In the future, we should
        # do this the right way using `importlib.resources`. For now, let's just
        # hardcode the relative path to the XML data directory.
        runconfig = f"runconfig_{wfname}{suf}.yaml"
        cmd = [f"time nisarqa {wfname}_qa {runconfig}"]
        try:
            self.distribrun(testdir, cmd, logfile=log, nisarimg=True, dataname=dataname,
                            loghdlrname=f'wfqa.{os.path.basename(testdir)}')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Workflow QA on test {testname} failed") from e

    def rslcqa(self, tests=None):
        if tests is None:
            tests = workflowtests['rslc'].items()
        for testname, dataname in tests:
            self.workflowqa("rslc", testname, dataname=dataname)

    def gslcqa(self, tests=None):
        if tests is None:
            tests = workflowtests['gslc'].items()
        for testname, dataname in tests:
            self.workflowqa("gslc", testname, dataname=dataname)

    def gcovqa(self, tests=None):
        if tests is None:
            tests = workflowtests['gcov'].items()
        for testname, dataname in tests:
            self.workflowqa("gcov", testname, dataname=dataname)

    def insarqa(self, tests=None):
        """
        Run QA for InSAR workflow using the NISAR distrib image.

        InSAR QA is a special case since the workflow name is not the product name.
        Also, the --quality flag in verify_gunw.py cannot be used at the moment since
        gunw file does not contain any science data.
        """
        wfname = "insar"
        if tests is None:
            tests = workflowtests['insar'].items()
        for testname, dataname in tests:
            testdir = os.path.abspath(pjoin(self.testdir, testname))
            # Run QA for each of the InSAR products.
            # QA validation and/or reports may be disabled for each individual product
            # type in the runconfig.
            # Note that the InSAR workflow doesn't always produce all output products --
            # if any products are not generated, QA should be disabled for them in the
            # runconfig file.
            for product in ['rifg', 'runw', 'gunw', 'roff', 'goff']:
                print(f"\nRunning workflow QA on test {testname} {product.upper()} product\n")
                qadir = pjoin(testdir, f"qa_insar")
                os.makedirs(qadir, exist_ok=True)
                log = pjoin(qadir,f"stdouterr.log")
                runconfig = f"runconfig_{wfname}.yaml"
                cmd = [f"time nisarqa {product}_qa {runconfig}"]
                try:
                    self.distribrun(testdir, cmd, logfile=log, nisarimg=True, dataname=dataname,
                                    loghdlrname=f'wfqa.{os.path.basename(testdir)}.{product}')
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Workflow QA on test {testname} {product.upper()} product failed\n") from e

    def end2endqa(self, tests=None):
        """
        Run QA on all end2end workflow test results for one pair of L0B input data, including RSLC, GSLC, GCOV,
        RIFG, RUNW, GUNW.

        """
        if tests is None:
            tests = workflowtests['end2end'].items()
        for testname, dataname in tests:
            for wfname in ['rslc', 'gslc', 'gcov']:
                for suf, descr in [('_ref', 'reference'), ('_sec', 'secondary')]:
                    self.workflowqa(
                        wfname,
                        testname,
                        dataname=dataname,
                        suf=suf,
                        description=f' {wfname.upper()} {descr} product',
                    )

            self.insarqa([(testname, dataname)])


    def minqa(self):
        """
        Only run qa for first test in each workflow
        """
        self.rslcqa(tests=list(workflowtests['rslc'].items())[:1])
        self.gslcqa(tests=list(workflowtests['gslc'].items())[:1])
        self.gcovqa(tests=list(workflowtests['gcov'].items())[:1])
        self.insarqa(tests=list(workflowtests['insar'].items())[:1])

    def docsbuild(self):
        """
        Build documentation using Doxygen + Sphinx
        """

        docdir = f"{blddir}/docs-output"
        sphx_src = f"{srcdir}/doc/sphinx"
        sphx_conf = f"{blddir}/doc/sphinx"
        sphx_dir = f"{docdir}/sphinx"
        sphx_cache = f"{sphx_dir}/_doctrees"
        sphx_html = f"{sphx_dir}/html"

        self.docker_run_dev(f"""
            PYTHONPATH={blddir}/packages/isce3/extensions \
               sphinx-build -q -b html -c {sphx_conf} -d {sphx_cache} {sphx_src} {sphx_html}
            doxygen doc/doxygen/Doxyfile
            """)

        subprocess.check_call(f"cp -r {self.projblddir}/docs-output .".split())

    def prdocs(self):
        """
        Deploy PR documentation to github pages
        This job's special; it deploys so it needs a GIT_OAUTH_TOKEN
        and a pull ID for the upload in the environment
        """

        auth = os.environ["RTBURNS_PAT"].strip()
        pr_id = os.environ["CHANGE_ID"].strip()
        build_url = os.environ["BUILD_URL"].strip()

        subprocess.check_call("bash -c".split() + [f"""
            set -e

            git clone -q --depth 1 --no-checkout \
                    https://{auth}@github-fn.jpl.nasa.gov/isce-3/pr-docs

            set -x

            cd pr-docs
            git reset &> /dev/null

            mv ../docs-output {pr_id}
            git add {pr_id}

            git config --local user.name  "gmanipon"
            git config --local user.email "gmanipon@jpl.nasa.gov"
            git commit -m "PR {pr_id} ({build_url})" \
                    && git push -q || echo "no changes committed"
        """])


    def tartests(self):
        """
        Tar up test directories for delivery
        """
        for workflow in workflowtests:
            for test in workflowtests[workflow]:
                print(f"\ntarring workflow test {test}\n")
                subprocess.check_call(f"tar cvz --exclude scratch*/* -f {test}.tar.gz {test}".split(), cwd=self.testdir)

    def push(self):
        """
        Push the (non-release) NISAR redistributable image to the 'docker-develop-local'
        and 'docker-stage-local' registries on artifactory.
        """
        # The name:tag of the NISAR distrib image created by `self.makedistrib_nisar()`.
        nisar_distrib_name = self.imgname()

        # The name:tag to give the image in the remote registry.
        # Release images are tagged 'gov/nasa/jpl/nisar/adt/nisar-adt/isce3:{release}'
        # where `release` is e.g. 'r3.2'.
        # In this case, we're pushing a non-release image, which should go in the same
        # directory but be tagged 'devel' instead of the release number.
        remote_tag = "gov/nasa/jpl/nisar/adt/nisar-adt/isce3:devel"

        # The hostname:port of the 'docker-develop-local' registry on artifactory.
        # Maps to
        # https://artifactory.jpl.nasa.gov:443/artifactory/docker-develop-local/.
        # This is the registry that the PGE team pulls from.
        develop_server = "cae-artifactory.jpl.nasa.gov:16001"

        # The hostname:port of the 'docker-stage-local' registry on artifactory.
        # Maps to https://artifactory.jpl.nasa.gov:443/artifactory/docker-stage-local/.
        # Pushing to this registry allows the image to be scanned for vulnerabilities.
        stage_server = "cae-artifactory.jpl.nasa.gov:16002"

        # Get remote registry credentials.
        # These are stored as secret credentials by the Jenkins server and exposed as a
        # string in '<username>:<password>' format via the `ARTIFACTORY_API_KEY` env
        # variable in the Jenkinsfile.
        try:
            username_password = os.environ["ARTIFACTORY_API_KEY"]
        except KeyError as exc:
            errmsg = dedent("""
                artifactory credentials not found

                If running via Jenkins, check the Jenkinsfile to ensure that secret
                credentials are correctly stored in the env variable
                `ARTIFACTORY_API_KEY`.

                If running locally, you must define an environment variable
                `ARTIFACTORY_API_KEY` that contains Artifactory credentials in
                '<username>:<password>' format.
            """).strip()
            raise RuntimeError(errmsg) from exc

        # Split string into username & password components.
        try:
            username, password = username_password.split(":")
        except ValueError as exc:
            errmsg = (
                "bad format for artifactory credentials: expected a string in"
                f" '<username>:<password>' format, got {username_password:!r}"
            )
            raise RuntimeError(errmsg) from exc

        # Push to both docker registries.
        for server in [develop_server, stage_server]:
            push_to_registry(
                image=nisar_distrib_name,
                server=server,
                username=username,
                password=password,
                tag=remote_tag,
            )
