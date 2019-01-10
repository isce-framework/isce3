#cython: language_level=3
#
# Author: Bryan V. Riel, Piyush Agram
# Copyright 2017
#

#Include basic types
#To address the binary incompatibility warning from cython
include "DateTime.pxd"
include "Ellipsoid.pxd"
include "LUT1d.pxd"
include "Orbit.pxd"
include "Poly2d.pxd"
include "Radar.pxd"
include "ImageMode.pxd"
include "Identification.pxd"

# Include the io extensions
include "pyGDAL.pxd"
include "pyRaster.pxd"
include "pyIH5.pxd"

# Include the core extensions
include "pyTimeDelta.pxd"
include "pyDateTime.pxd"
include "pyEulerAngles.pxd"
include "pyQuaternion.pxd"
include "pyBasis.pxd"
include "pyDoppler.pxd"
include "pyEllipsoid.pxd"
include "pyInterpolator.pxd"
include "pyPeg.pxd"
include "pyPegtrans.pxd"
include "pyPosition.pxd"
include "pyLinAlg.pxd"
include "pyLUT1d.pxd"
include "pyOrbit.pxd"
include "pyPoly1d.pxd"
include "pyPoly2d.pxd"

# Include the radar extensions
include "pyRadar.pxd"

# Include the product extensions
include "pyImageMode.pxd"
include "pyIdentification.pxd"
include "pyMetadata.pxd"
include "pyComplexImagery.pxd"
include "pyProduct.pxd"

# Include the image extensions
include "pyResampSlc.pxd"

# Include the geometry extensions
include "pyTopo.pxd"
include "pyGeo2rdr.pxd"

# end of file
