from .imgset import ImageSet, docker, docker_info, thisdir
import subprocess

# nb: pyre 1.9.8 needs monkey patching to remove <sys/sysctl.h>
# see https://github.com/pyre/pyre/issues/54

class AlpineImageSet(ImageSet):
    def __init__(self, **kwargs):
        super().__init__("alpine", **kwargs)
        self.build_args += " --build-arg conda_prefix=/opt/conda"
        self.cmake_defs.update(
            {
                "CPACK_INCLUDE_TOPLEVEL_DIRECTORY": "NO",
                "CMAKE_INSTALL_LIBDIR": "lib",
                "WITH_CUDA": "NO",
                "ISCE3_FETCH_EIGEN": "NO",
                "Eigen3_DIR": "/usr/share/cmake/Modules",
            }
        )
        self.cpack_generator = "TGZ"

    def makedistrib(self):
        """
        install package to redistributable isce3 docker image
        """
        subprocess.check_call(f"cp isce3.tar.gz \
                {thisdir}/{self.name}/distrib/isce3.tar.gz".split())

        # Squashing requires experimental features
        squash_arg = "--squash" if "Experimental: true" in docker_info else ""

        cmd = f"{docker} build {self.build_args} {squash_arg} \
                    {thisdir}/{self.name}/distrib -t nisar-adt/isce3:{self.name}"
        subprocess.check_call(cmd.split())
