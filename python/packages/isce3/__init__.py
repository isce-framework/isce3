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
from . import ionosphere
from . import math
from . import polsar
from . import product
from . import signal
from . import splitspectrum
from . import unwrap

# check for cuda
if hasattr(extisce3, "cuda"):
    from . import cuda
