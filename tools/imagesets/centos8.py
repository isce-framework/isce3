from .imgset import ImageSet

class Centos8EpelImageSet(ImageSet):
    def __init__(self, **kwargs):
        super().__init__("centos8epel", **kwargs)
        self.cmake_defs.update({
            "ISCE3_FETCH_EIGEN": "YES",
        })
