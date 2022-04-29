from . import alpine, centos7, oracle8
from .imgset import projsrcdir

imagesets = {
    "alpine": alpine.AlpineImageSet,
    "centos7conda": centos7.Centos7CondaImageSet,
    "oracle8conda": oracle8.Oracle8CondaImageSet,
}
