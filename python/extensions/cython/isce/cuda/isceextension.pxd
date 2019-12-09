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
include "LUT2d.pxd"
include "Orbit.pxd"
include "Poly2d.pxd"

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
include "pyEllipsoid.pxd"
include "pyInterpolator.pxd"
include "pyPeg.pxd"
include "pyPegtrans.pxd"
include "pyLUT1d.pxd"
include "pyLUT2d.pxd"
include "pyOrbit.pxd"
include "pyPoly1d.pxd"
include "pyPoly2d.pxd"

# Include the product extensions
include "pyMetadata.pxd"
include "pyProduct.pxd"
include "pySwath.pxd"
include "pyProcessingInformation.pxd"

# Include the image extensions
include "pyResampSlc.pxd"

# Include the geometry extensions
include "pyTopo.pxd"
include "pyGeo2rdr.pxd"

# Include the signal extensions
include "Crossmul.pxd"

# end of file
