# -*- Makefile -*-

# global project settings
include isce.def
# package isce/geometry
PACKAGE = isce/geometry

# the list of sources
PROJ_SRCS = \
    DEMInterpolator.cpp \
    Geometry.cpp \
    Topo.cpp \

# products
# the library
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(PROJECT_MAJOR).$(PROJECT_MINOR).$(EXT_SO)
EXPORT_LIBS = $(PROJ_DLL)
# the headers
EXPORT_PKG_HEADERS = \
    DEMInterpolator.h \
    Geometry.h \
    Topo.h \

# standard targets
all: $(PROJ_DLL) export

export:: export-package-headers export-libraries

# configuration
# the extension of the c++ sources
EXT_CXX = cpp

# the private build location
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)-$(PROJECT_MAJOR).$(PROJECT_MINOR)/lib

# end-of-file
