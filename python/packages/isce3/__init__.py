# pull the bindings
from .ext import extisce3

# Inherit dunder attributes from pybind11 bindings
__doc__ = extisce3.__doc__
__version__ = extisce3.__version__

# export the subpackages
from . import antenna
from . import container
from . import core
from . import focus
from . import geocode
from . import geometry
from . import geogrid
from . import image
from . import io
from . import matchtemplate
from . import math
from . import polsar
from . import product
from . import signal
from . import solid_earth_tides
from . import splitspectrum
from . import unwrap

from . import atmosphere
# Need to import `cal` after the other submodules that it depends on have been added as
# attributes to `isce3` above. (If you try to import it in alphabetical order you will
# get an AttributeError.)
from . import cal

# check for cuda
if hasattr(extisce3, "cuda"):
    from . import cuda
