from .imgset import ImageSet

class Oracle8CondaImageSet(ImageSet):
    def __init__(self, **kwargs):
        super().__init__("oracle8conda", **kwargs)
        self.build_args += " --build-arg conda_prefix=/opt/conda"
        self.cmake_defs.update({
            # Set DESTDIR so that paths outside of conventional prefix
            # (i.e. python components en route to conda prefix)
            # are coerced to install regardless
            "CPACK_SET_DESTDIR": "YES",

            # Tell rpm to use conda's python for installation,
            # overriding the system python
            "CPACK_RPM_SPEC_MORE_DEFINE":
                '"%define __python $CONDA_PREFIX/bin/python"',

            # Point to nonstandard git location
            "GIT_EXECUTABLE": "/opt/conda/bin/git",
        })
