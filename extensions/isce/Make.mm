# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# project global settings
include isce.def

# the package
PACKAGE = extensions
# the module
MODULE = isceextension
# use a tmp directory that knows the name of the module
PROJ_TMPDIR = $(BLD_TMPDIR)/extensions/$(PROJECT)/$(MODULE)

# project settings: do not remove core directory (core usually refers core dump file)
# filter-out info at: https://www.gnu.org/software/make/manual/html_node/index.html
PROJ_TIDY := ${filter-out core, $(PROJ_TIDY)}

# my includes
PROJ_CXX_INCLUDES = $(EXPORT_INCDIR)
# point to the location of my libraries
PROJ_LCXX_LIBPATH = $(BLD_LIBDIR)
# link against these
PROJ_LIBRARIES = -lisce.$(PROJECT_MAJOR).$(PROJECT_MINOR) -ljournal

# the sources
MODULE_CYTHON_PYX = \

# the headers
MODULE_CYTHON_PXD = \

# use cython to build a python extension
include std-cython.def

# end of file
