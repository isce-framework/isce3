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
        return s

include "pyTimeDelta.pyx"
include "pyDateTime.pyx"
include "pyAttitude.pyx"
include "pyBasis.pyx"
include "pyDoppler.pyx"
include "pyEllipsoid.pyx"
#include "pyInterpolator.pyx"
include "pyPeg.pyx"
include "pyPegtrans.pyx"
include "pyPosition.pyx"
include "pyMetadata.pyx"
include "pyLinAlg.pyx"
include "pyOrbit.pyx"
include "pyPoly1d.pyx"
include "pyPoly2d.pyx"
include "pyRaster.pyx"
