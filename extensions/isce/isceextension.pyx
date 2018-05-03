#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

# Define a helper function to convert Python strings/bytes to bytes
def pyStringToBytes(s):
    if isinstance(s, str):
        return s.encode('utf-8')
    elif isinstance(s, bytes):
        return s
    else:
        raise ValueError('Input Python string not str or bytes')

# Include the core extensions
include "core/pyTimeDelta.pyx"
include "core/pyDateTime.pyx"
include "core/pyAttitude.pyx"
include "core/pyBasis.pyx"
include "core/pyDoppler.pyx"
include "core/pyEllipsoid.pyx"
#include "core/pyInterpolator.pyx"
include "core/pyPeg.pyx"
include "core/pyPegtrans.pyx"
include "core/pyPosition.pyx"
include "core/pyMetadata.pyx"
include "core/pyLinAlg.pyx"
include "core/pyOrbit.pyx"
include "core/pyPoly1d.pyx"
include "core/pyPoly2d.pyx"
include "core/pyRaster.pyx"
include "core/pyResampSlc.pyx"

# Include the geometry extensions
include "geometry/pyTopo.pyx"
include "geometry/pyGeo2rdr.pyx"

# end of file
