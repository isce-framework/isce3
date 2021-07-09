import os, subprocess, sys, shutil, stat, logging, shlex, getpass
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
    with pipe.stdout:
        for line in iter(pipe.stdout.readline, b''): # b'\n'-separated lines
            decoded = line.decode("utf-8")
            # remove newline character so the log does not contain extra blank lines
            if str.endswith(decoded, '\n'):
                decoded = decoded[:-1]
            logger.info(decoded)
    ret = pipe.poll()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmdstr)

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
        if self.imgtag:
            return f"nisar-adt/isce3{repomod}:{self.imgtag}"
        else:
            if tagmod != "":
                tagmod = "-" + tagmod
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
                    {thisdir}/{self.name}/distrib -t {self.imgname()}"
        subprocess.check_call(cmd.split())


    def makedistrib_nisar(self):
        """
        Install package to redistributable isce3 docker image with nisar qa and 
        noise estimator caltool
        """

        build_args = f"--build-arg distrib_img={self.imgname()} \
                       --build-arg GIT_OAUTH_TOKEN={os.environ.get('GIT_OAUTH_TOKEN').strip()}"
        
        cmd = f"{docker} build {build_args} \
                {thisdir}/{self.name}/distrib_nisar -t {self.imgname(tagmod='nisar')}"
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

        mindata = ["L0B_RRSD_REE1",
                   "L0B_RRSD_REE_NOISEST1",
                   "L0B_RRSD_REE_PTA1",
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
            img = self.imgname(tagmod="nisar")
        else:
            img = self.imgname()

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

    def workflowtest(self, wfname, testname, dataname, pyname, arg=""): # hmmmmmmmmm
        """
        Run the specified workflow test using the distrib image.
        
        Parameters
        -------------
        wfname : str
            Workflow name (e.g. "rslc")
        testname : str
            Workflow test name (e.g. "RSLC_REE1")
        dataname : str
            Test input data (e.g. "L0B_RRSD_REE1")
        pyname : str
            Name of the isce3 module to execute (e.g. "pybind_nisar.workflows.focus")
        arg : str, optional
            Additional command line argument(s) to pass to the workflow
        """
        print(f"\nRunning workflow test {testname}\n")
        testdir = os.path.abspath(pjoin(self.testdir, testname))
        os.makedirs(pjoin(testdir, f"output_{wfname}"), exist_ok=True)
        os.makedirs(pjoin(testdir, f"scratch_{wfname}"), exist_ok=True)
        # copy test runconfig to test directory
        shutil.copyfile(pjoin(runconfigdir, f"{testname}.yaml"), 
                        pjoin(testdir, f"runconfig_{wfname}.yaml"))
        log = pjoin(testdir, f"output_{wfname}", "stdouterr.log")
        cmd = [f"time python3 -m {pyname} {arg} runconfig_{wfname}.yaml"]
        try:
            self.distribrun(testdir, cmd, logfile=log, dataname=dataname,
                            loghdlrname=f'wftest.{os.path.basename(testdir)}')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Workflow test {testname} failed") from e

    def rslctest(self, tests=None):
        if tests is None:
            tests = workflowtests['rslc'].items()
        for testname, dataname in tests:
            self.workflowtest("rslc", testname, dataname, "pybind_nisar.workflows.focus")
            
    def gslctest(self, tests=None):
        if tests is None:
            tests = workflowtests['gslc'].items()
        for testname, dataname in tests:
            self.workflowtest("gslc", testname, dataname, "pybind_nisar.workflows.gslc")
    
    def gcovtest(self, tests=None):
        if tests is None:
            tests = workflowtests['gcov'].items()
        for testname, dataname in tests:
            self.workflowtest("gcov", testname, dataname, "pybind_nisar.workflows.gcov")

    def insartest(self, tests=None):
        if tests is None:
            tests = workflowtests['insar'].items()
        for testname, dataname in tests:
            self.workflowtest("insar", testname, dataname, "pybind_nisar.workflows.insar", arg="--restart")

    def noisesttest(self, tests=None):
        if tests is None:
            tests = workflowtests['noisest'].items()
        for testname, dataname in tests:
            print(f"\nRunning CalTool noise estimate test {testname}\n")
            testdir = os.path.abspath(pjoin(self.testdir, testname))
            os.makedirs(pjoin(testdir, f"output_noisest"), exist_ok=True)
            log = pjoin(testdir, f"output_noisest", "stdouterr.log")
            cmd = [f"""time noise_evd_estimate.py -i input_{dataname}/{workflowdata[dataname][0]} \
                                                  -r -c 10 -o output_noisest/noise_est_output_bcal.txt"""]
            try: 
                self.distribrun(testdir, cmd, logfile=log, dataname=dataname, nisarimg=True,
                                loghdlrname=f'wftest.{os.path.basename(testdir)}')
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"CalTool noise estimate test {testname} failed") from e

    def ptatest(self, tests=None):
        if tests is None:
            tests = workflowtests['pta'].items()
        for testname, dataname in tests:
            print(f"\nRunning CalTool point target analyzer test {testname}\n")
            testdir = os.path.abspath(pjoin(self.testdir, testname))
            os.makedirs(pjoin(testdir, f"output_pta"), exist_ok=True)
            log = pjoin(testdir, f"output_pta", "stdouterr.log")
            cmd = [f"""time python -m pybind_nisar.workflows.point_target_info \
                                   input_{dataname}/{workflowdata[dataname][0]} 512 256 256 \
                                   --fs-bw-ratio 2 --mlobe-nulls 2 --search-null > output_pta/pta.json"""]
            try:
                self.distribrun(testdir, cmd, logfile=log, dataname=dataname, nisarimg=True,
                                loghdlrname=f'wftest.{os.path.basename(testdir)}')
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"CalTool point target analyzer tool test {testname} failed") from e

    def mintests(self):
        """
        Only run first test from each workflow
        """
        self.rslctest(tests=list(workflowtests['rslc'].items())[:1])
        self.gslctest(tests=list(workflowtests['gslc'].items())[:1])
        self.gcovtest(tests=list(workflowtests['gcov'].items())[:1])
        self.insartest(tests=list(workflowtests['insar'].items())[:1])
        self.noisesttest(tests=list(workflowtests['noisest'].items())[:1])
        self.ptatest(tests=list(workflowtests['pta'].items())[:1])

    def workflowqa(self, wfname, testname):
        """
        Run QA and CF compliance checking for the specified workflow using the NISAR distrib image.
        
        Parameters
        -------------
        wfname : str
            Workflow name (e.g. "rslc")
        testname: str
            Workflow test name (e.g. "RSLC_REE1")
        """
        print(f"\nRunning workflow QA on test {testname}\n")
        testdir = os.path.abspath(pjoin(self.testdir, testname))
        os.makedirs(pjoin(testdir, f"qa_{wfname}"), exist_ok=True)
        log = pjoin(testdir, f"qa_{wfname}", "stdouterr.log")
        cmd = [f"time cfchecks.py output_{wfname}/{wfname}.h5",
               f"""time verify_{wfname}.py --fpdf qa_{wfname}/graphs.pdf \
                    --fhdf qa_{wfname}/stats.h5 --flog qa_{wfname}/qa.log --validate \
                    --quality output_{wfname}/{wfname}.h5"""]
        try:
            self.distribrun(testdir, cmd, logfile=log, nisarimg=True,
                            loghdlrname=f'wfqa.{os.path.basename(testdir)}')
        except subprocess.CalledProcessError as e:
            if testname in ['rslc_REE2', 'gslc_UAVSAR_SanAnd_05518_12018_000_120419_L090_CX_143_03',
                            'gslc_UAVSAR_SanAnd_05518_12128_008_121105_L090_CX_143_02',
                            'gslc_UAVSAR_Snjoaq_14511_18034_014_180720_L090_CX_143_02',
                            'gslc_UAVSAR_Snjoaq_14511_18044_015_180814_L090_CX_143_02']:
                # Don't exit workflow QA when it fails due to known memory issue reading large files
                # QA tests that fail on both nisar-adt-dev-1 and nisar-adt-dev-2:
                #     rslc_REE2
                #     gslc_UAVSAR_Snjoaq_14511_18034_014_180720_L090_CX_143_02
                #     gslc_UAVSAR_Snjoaq_14511_18044_015_180814_L090_CX_143_02
                # Extra QA tests that fail on nisar-adt-dev-2:
                #     gslc_UAVSAR_SanAnd_05518_12018_000_120419_L090_CX_143_03
                #     gslc_UAVSAR_SanAnd_05518_12128_008_121105_L090_CX_143_02
                print(f"Known failure running workflow QA on test {testname}")
            else:
                raise RuntimeError(f"Workflow QA on test {testname} failed") from e

    def rslcqa(self, tests=None):
        if tests is None:
            tests = workflowtests['rslc'].keys()
        for testname in tests:
            self.workflowqa("rslc", testname)

    def gslcqa(self, tests=None):
        if tests is None:
            tests = workflowtests['gslc'].keys()
        for testname in tests:
            self.workflowqa("gslc", testname)

    def gcovqa(self, tests=None):
        if tests is None:
            tests = workflowtests['gcov'].keys()
        for testname in tests:
            self.workflowqa("gcov", testname)

    def insarqa(self, tests=None):
        """
        Run QA and CF compliance checking for InSAR workflow using the NISAR distrib image.

        InSAR QA is a special case since the workflow name is not the product name. 
        Also, the --quality flag in verify_gunw.py cannot be used at the moment since
        gunw file does not contain any science data.
        """
        wfname = "insar"
        if tests is None:
            tests = workflowtests['insar'].keys()
        for testname in tests:
            testdir = os.path.abspath(pjoin(self.testdir, testname))
            # run QA for each of the InSAR products
            for product in ['rifg', 'runw', 'gunw']:
                print(f"\nRunning workflow QA on InSAR test {testname} product {product.upper()}\n")
                qadir = pjoin(testdir, f"qa_{product}")
                os.makedirs(qadir, exist_ok=True)
                log = pjoin(qadir,f"stdouterr.log")
                cmd = [f"time cfchecks.py output_{wfname}/{product.upper()}_product.h5"]
                if product == 'gunw':
                    cmd.append(f"""time verify_gunw.py --fpdf qa_{product}/graphs.pdf \
                                       --fhdf qa_{product}/stats.h5 --flog qa_{product}/qa.log --validate \
                                       output_{wfname}/{product.upper()}_product.h5""")
                try:
                    self.distribrun(testdir, cmd, logfile=log, nisarimg=True, 
                                    loghdlrname=f'wfqa.{os.path.basename(testdir)}.{product}')
                except subprocess.CalledProcessError as e:
                    if product == 'gunw':
                        raise RuntimeError(f"Workflow QA on InSAR test {testname} product {product.upper()} failed\n") from e
                    else:
                        # do not exit since CF checker errors are expected
                        print(f"Found known errors running CF Checker on InSAR test {testname} product {product.upper()}\n")
       
    def minqa(self):
        """
        Only run qa for first test in each workflow
        """
        self.rslcqa(tests=list(workflowtests['rslc'].keys())[:1])
        self.gslcqa(tests=list(workflowtests['gslc'].keys())[:1])
        self.gcovqa(tests=list(workflowtests['gcov'].keys())[:1])
        self.insarqa(tests=list(workflowtests['insar'].keys())[:1])

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
        pr_id = os.environ["ghprbPullId"].strip()
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
                subprocess.check_call(f"tar cvzf {test}.tar.gz {test}".split(), cwd=self.testdir)
