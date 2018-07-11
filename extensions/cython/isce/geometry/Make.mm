# -*- Makefile -*-
#

# project defaults
include isce.def

# the package
PACKAGE = extensions
# the module
MODULE = iscegeometry
# use a tmp directory that knows the name of the module
PROJ_TMPDIR = $(BLD_TMPDIR)/extensions/$(PROJECT)/$(MODULE)

# my includes
PROJ_CXX_INCLUDES = $(EXPORT_INCDIR)
CYTHON_FLAGS = -I../core
# point to the location of my libraries
PROJ_LCXX_LIBPATH = $(BLD_LIBDIR)
# link against these
PROJ_LIBRARIES = -lisce.$(PROJECT_MAJOR).$(PROJECT_MINOR) -ljournal

# the sources
MODULE_CYTHON_PYX = \
    pyTopo.pyx \

# the headers
MODULE_CYTHON_PXD = \
    Topo.pxd \
    SerializeGeometry.pxd \

# use cython to build a python extension
include std-cython.def

# end of file
