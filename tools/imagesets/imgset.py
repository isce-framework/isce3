import os, subprocess, sys, shutil, stat, logging
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

def run_with_logging(cmd, logger, checkrun=True, printlog=True):
    """
    Run command as a subprocess and log the standard streams (stdout & stderr) to
    the specified logger.

    Parameters
    -------------
    cmd : list
        Command in list of strings
    logger : logger
        Python logger to log output, could be to standard out or a file
    checkrun : Boolean (optional)
        Exit if run returns non-zero status
    """
    logger.propagate = printlog
    # save command
    logger.info("++ " + subprocess.list2cmdline(cmd) + "\n")
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with pipe.stdout:
        for line in iter(pipe.stdout.readline, b''): # b'\n'-separated lines
            decoded = line.decode("utf-8")
            # remove newline character so the log does not contain extra blank lines
            if str.endswith(decoded, '\n'):
                decoded = decoded[:-1]
            logger.info(decoded)
    rc = pipe.poll()
    if checkrun and (rc != 0):
        sys.exit(rc)

# A set of docker images suitable for building and running isce3
class ImageSet:
    # images in the set, in the order they should be built
    imgs = [
        "runtime",
        "dev",
    ]

    cpack_generator = "RPM"

    def imgname(self, img):
        return f"nisar-adt/isce3dev:{self.name}-{img}" # labels used for above images

    def docker_run(self, img, cmd):
        runcmd = f"{docker} run {self.run_args} --rm -i {self.tty} {self.imgname(img)} bash -ci"
        subprocess.check_call(runcmd.split() + [cmd])

    def docker_run_dev(self, cmd):
        """
        Shortcut since most commands just run in the devel image
        """
        self.docker_run("dev", cmd)

    def __init__(self, name, *, projblddir, printlog=False):
        """
        A set of docker images for building and testing isce3/nisar distributables.
        
        Parameters
        ----------
        name : str
            Name of the image set (e.g. "centos7", "alpine"). Used for tagging images
        projblddir : str
            Path to the binary directory on the host where build artifacts are written
        printlog : boolean
            Print workflow test and qa logs to console in real-time
        """
        self.name = name
        self.projblddir = projblddir
        self.datadir = projblddir + "/workflow_testdata_tmp/data"
        self.testdir = projblddir + "/workflow_testdata_tmp/test"
        self.build_args = f'''
            --network=host
        ''' + " ".join(f"--build-arg {x}_img={self.imgname(x)}" for x in self.imgs)

        self.cmake_defs = {
            "WITH_CUDA": "YES",
            "ISCE3_FETCH_EIGEN": "NO",
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
            cmd = f"{docker} build {self.build_args} {thisdir}/{self.name}/{img} -t {self.imgname(img)}"
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
                    {thisdir}/{self.name}/distrib -t nisar-adt/isce3:{self.name}"
        subprocess.check_call(cmd.split())


    def makedistrib_nisar(self):
        """
        Install package to redistributable isce3 docker image with nisar qa and 
        noise estimator caltool
        """

        build_args = f"--build-arg distrib_img=nisar-adt/isce3:{self.name} \
                       --build-arg GIT_OAUTH_TOKEN={os.environ.get('GIT_OAUTH_TOKEN')}"
        
        cmd = f"{docker} build {build_args} \
                {thisdir}/{self.name}/distrib_nisar -t nisar-adt/isce3:{self.name}-nisar"
        subprocess.check_call(cmd.split())


    def fetchdata(self):
        """
        Fetch workflow testing data from Artifactory
        TODO use a properly cached download e.g. via DVC
        """


        # Download files, preserving relative directory hierarchy
        for testname, fetchfiles in workflowdata.items():
            wfdatadir = f"{self.datadir}/{testname}"
            os.makedirs(wfdatadir, exist_ok=True)
            for fname in fetchfiles:
                url = f"{art_base}/{testname}/{fname}"
                print("Fetching file:", url)
                subprocess.check_call(f"curl -f --create-dirs -o {fname} -O {url} ".split(),
                                      cwd = wfdatadir)

    def distribrun(self, testdir, cmd, logfile=None, dataname=None, nisarimg=False, checkrun=True):
        """
        Run a command in the distributable image

        Parameters
        -------------
        testdir : str
            Test directory to run Docker command in
        cmd : str
            Command to run inside Docker
        logfile : str, optional
            File name (relative to testdir) of log file for saving standard out and standard error
        dataname : str or list, optional
            Test input data as str or list (e.g. "L0B_RRSD_REE1", ["L0B_RRSD_REE1", "L0B_RRSD_REE2")
        nisarimg : boolean, optional
            Use NISAR distributable image
        """
        # save stdout and stderr to logfile if specified
        logger = logging.getLogger(name=f'workflow.{os.path.basename(testdir)}')
        if logfile is None:
            hdlr = logging.StreamHandler(sys.stdout)
        else:
            hdlr = logging.FileHandler(pjoin(testdir, logfile), mode='w')
        logger.addHandler(hdlr)
       
        if nisarimg:
            tag = self.name + "-nisar"
        else:
            tag = self.name
        img = 'nisar-adt/isce3:' + tag

        datamount = ""
        if dataname is not None:
            if type(dataname) is not list:
                dataname = [dataname]
            for data in dataname:
                datadir = os.path.abspath(pjoin(self.datadir, data))          
                datamount += f"-v {datadir}:{container_testdir}/input_{data}:ro "

        runcmd = f"{docker} run \
          -v {testdir}:{container_testdir} {datamount} \
          -w {container_testdir} \
          -u {os.getuid()}:{os.getgid()} \
          --rm -i {self.tty} {img} sh -ci"  
        run_with_logging(runcmd.split() + [cmd], logger, checkrun=checkrun, printlog=self.printlog)

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
        script = f"""
            time python3 -m {pyname} {arg} runconfig_{wfname}.yaml
            """
        self.distribrun(testdir, script, logfile=log, dataname=dataname)

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
            print(f"\nRunning calTool noise estimate test for {testname}\n")
            testdir = os.path.abspath(pjoin(self.testdir, testname))
            os.makedirs(pjoin(testdir, f"output_noiseest"), exist_ok=True)
            log = pjoin(testdir, f"output_noiseest", "stdouterr.log")
            script = f"""
                time noise_evd_estimate.py -i input_{dataname}/{workflowdata[dataname][0]} -r
                """
            self.distribrun(testdir, script, logfile=log, dataname=dataname, nisarimg=True)

    def mintests(self):
        """
        Only run first test from each workflow
        """
        self.rslctest(tests=list(workflowtests['rslc'].items())[:1])
        self.gslctest(tests=list(workflowtests['gslc'].items())[:1])
        self.gcovtest(tests=list(workflowtests['gcov'].items())[:1])
        self.insartest(tests=list(workflowtests['insar'].items())[:1])
        self.noisesttest(tests=list(workflowtests['noisest'].items())[:1])

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
        verifycmd = f"""time verify_{wfname}.py --fpdf qa_{wfname}/graphs.pdf \
                --fhdf qa_{wfname}/stats.h5 --flog qa_{wfname}/qa.log --validate \
                --quality output_{wfname}/{wfname}.h5"""
        # prepare script, removing extra white space
        script = f"""
            time cfchecks.py output_{wfname}/{wfname}.h5
            """ + " ".join(verifycmd.split())
        self.distribrun(testdir, script, logfile=log, nisarimg=True)

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
                script = f"""
                    time cfchecks.py output_{wfname}/{product.upper()}_gunw.h5
                    """
                if product == 'gunw':
                    verifycmd = f"""time verify_gunw.py --fpdf qa_{product}/graphs.pdf \
                        --fhdf qa_{product}/stats.h5 --flog qa_{product}/qa.log --validate \
                        output_{wfname}/{product.upper()}_gunw.h5"""
                    # add to script while removing extra white space
                    script += " ".join(verifycmd.split())
                    self.distribrun(testdir, script, logfile=log, nisarimg=True)
                else:
                    self.distribrun(testdir, script, logfile=log, nisarimg=True, checkrun=False)
       
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

