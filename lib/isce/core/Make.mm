# -*- Makefile -*-

# global project settings
include isce.def
# package isce/core
PACKAGE = isce/core

# the list of sources
PROJ_SRCS = \
    Baseline.cpp \
    DateTime.cpp \
    Ellipsoid.cpp \
    Interpolator.cpp \
    LUT2d.cpp \
    LinAlg.cpp \
    Metadata.cpp \
    Orbit.cpp \
    Peg.cpp \
    Pegtrans.cpp \
    Poly1d.cpp \
    Poly2d.cpp \
    Position.cpp \
    Projections.cpp \

# products
# the library
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(PROJECT_MAJOR).$(PROJECT_MINOR).$(EXT_SO)
EXPORT_LIBS = $(PROJ_DLL)
# the headers
EXPORT_PKG_HEADERS = \
    Baseline.h \
    Constants.h \
    DateTime.h \
    Ellipsoid.h \
    Interpolator.h \
    LUT2d.h \
    LinAlg.h \
    Metadata.h \
    Orbit.h \
    Peg.h \
    Pegtrans.h \
    Poly1d.h \
    Poly2d.h \
    Position.h \
    Projections.h \

# standard targets
all: $(PROJ_DLL) export

export:: export-package-headers export-libraries

# configuration
# the extension of the c++ sources
EXT_CXX = cpp

# the private build location
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)-$(PROJECT_MAJOR).$(PROJECT_MINOR)/lib


# end-of-file
