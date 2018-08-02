# -*- Makefile -*-
#

# project defaults
include isce.def

# the package
PACKAGE = extensions
# the module
MODULE = iscecore
# use a tmp directory that knows the name of the module
PROJ_TMPDIR = $(BLD_TMPDIR)/extensions/$(PROJECT)/$(MODULE)

# my includes
PROJ_CXX_INCLUDES = $(EXPORT_INCDIR)
# point to the location of my libraries
PROJ_LCXX_LIBPATH = $(BLD_LIBDIR)
# link against these
PROJ_LIBRARIES = -lisce.$(PROJECT_MAJOR).$(PROJECT_MINOR) -ljournal

# the sources
MODULE_CYTHON_PYX = \
    pyAttitude.pyx \
    pyBasis.pyx \
    pyDateTime.pyx \
    pyDoppler.pyx \
    pyEllipsoid.pyx \
    pyLinAlg.pyx \
    pyMetadata.pyx \
    pyOrbit.pyx \
    pyPeg.pyx \
    pyPegtrans.pyx \
    pyPoly1d.pyx \
    pyPoly2d.pyx \
    pyPosition.pyx \
    pyTimeDelta.pyx \

# the headers
MODULE_CYTHON_PXD = \
    Attitude.pxd \
    Basis.pxd \
    DateTime.pxd \
    Doppler.pxd \
    Ellipsoid.pxd \
    LinAlg.pxd \
    Metadata.pxd \
    Orbit.pxd \
    Peg.pxd \
    Pegtrans.pxd \
    Poly1d.pxd \
    Poly2d.pxd \
    Position.pxd \
    Serialization.pxd \
    TimeDelta.pxd \

# use cython to build a python extension
include std-cython.def

# end of file
