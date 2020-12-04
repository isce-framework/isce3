import os, subprocess, sys, shutil, stat
from .workflowdata import rslctestdict, gslctestdict, gcovtestdict
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

    def __init__(self, name, *, projblddir):
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
        Install package to redistributable isce3 docker image with nisar qa and caltools
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

        # see no evil
        from .workflowdata import workflowdata

        # Download files, preserving relative directory hierarchy
        for (testname, fetchfiles) in workflowdata:
            wfdatadir = f"{self.datadir}/{testname}"
            os.makedirs(wfdatadir, exist_ok=True)
            for fname in fetchfiles:
                url = f"{art_base}/{testname}/{fname}"
                print("Fetching file:", url)
                subprocess.check_call(f"curl -f --create-dirs -o {fname} -O {url} ".split(),
                                      cwd = wfdatadir)

    def distribrun(self, testdir, cmd, log=None, dataname=None, nisarimg=False):
        """
        Run a command in the distributable image

        Parameters
        -------------
        testdir : str
            Test directory to run Docker command in
        cmd : str
            Command to run inside Docker
        log : str, optional
            Path of log file for saving standard out and standard error
        dataname : str, optional
            Test input data (e.g. "L0B_RRSD_REE1")
        nisarimg : boolean, optional
            Use NISAR distributable image
        """
        # save stdout and stderr to logfile if specified
        if log is not None:
            logpath = pjoin(testdir, log)
            logfh = open(logpath, "w")
        else:
            logfh = None

        if nisarimg:
            tag = self.name + "-nisar"
        else:
            tag = self.name

        if dataname is not None:
            datadir = os.path.abspath(pjoin(self.datadir, dataname))          
            datamount = f"--mount type=bind,source={datadir},target={container_testdir}/input_{dataname}"
        else:
            datamount = ""

        runcmd = f"{docker} run \
          --mount type=bind,source={testdir},target={container_testdir} {datamount} \
          -w {container_testdir} \
          -u {os.getuid()}:{os.getgid()} \
          --rm -i {self.tty} nisar-adt/isce3:{tag} sh -ci"  
        if log is not None: 
            # save command in logfile
            logfh.write("++ " + subprocess.list2cmdline(runcmd.split() + [cmd]) + "\n")
            logfh.flush()
        subprocess.check_call(runcmd.split() + [cmd], stdout=logfh, stderr=subprocess.PIPE)

        if log is not None:
            logfh.close()
            # print log to screen for easy viewing
            with open(logpath, "r") as logfh:
                print(logfh.read())

    def workflowtest(self, wfname, testname, dataname, pyname): # hmmmmmmmmm
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
        """
        testdir = os.path.abspath(pjoin(self.testdir, testname))
        os.makedirs(pjoin(testdir, f"output_{wfname}"), exist_ok=True)
        os.makedirs(pjoin(testdir, f"scratch_{wfname}"), exist_ok=True)
        # copy test runconfig to test directory
        shutil.copyfile(pjoin(runconfigdir, f"{testname}.yaml"), 
                        pjoin(testdir, f"runconfig_{wfname}.yaml"))
        log = pjoin(testdir, f"output_{wfname}", "stdouterr.log")
        script = f"""
            python3 -m {pyname} runconfig_{wfname}.yaml
            """
        self.distribrun(testdir, script, log, dataname=dataname)

    def rslctest(self):
        for testname, dataname in rslctestdict.items():
            self.workflowtest("rslc", testname, dataname, "pybind_nisar.workflows.focus")
            
    def gslctest(self):
        for testname, dataname in gslctestdict.items():
            self.workflowtest("gslc", testname, dataname, "pybind_nisar.workflows.gslc")
    
    def gcovtest(self):
        for testname, dataname in gcovtestdict.items():
            self.workflowtest("gcov", testname, dataname, "pybind_nisar.workflows.gcov")

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
        testdir = os.path.abspath(pjoin(self.testdir, testname))
        os.makedirs(pjoin(testdir, f"qa_{wfname}"), exist_ok=True)
        log = pjoin(testdir, f"qa_{wfname}", "stdouterr.log")
        script = f"""
            time verify_{wfname}.py --fpdf qa_{wfname}/graphs.pdf \
                --fhdf qa_{wfname}/stats.h5 --flog qa_{wfname}/qa.log --validate \
                --quality output_{wfname}/{wfname}.h5
            time cfchecks.py output_{wfname}/{wfname}.h5
            echo ""
            """
        self.distribrun(testdir, script, log, nisarimg=True)

    def rslcqa(self):
        for testname in rslctestdict:
            self.workflowqa("rslc", testname)
    def gslcqa(self):
        for testname in gslctestdict:
            self.workflowqa("gslc", testname)
    def gcovqa(self):
        for testname in gcovtestdict:
            self.workflowqa("gcov", testname)

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
