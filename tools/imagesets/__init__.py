from . import alpine, centos7
from .imgset import projsrcdir

imagesets = {
    "alpine": alpine.AlpineImageSet,
    "centos7conda": centos7.Centos7CondaImageSet,
}
