# -*- Makefile -*-

# global project settings
include isce.def
# package isce/core
PACKAGE = isce/core

# the list of sources
PROJ_SRCS = \
    Baseline.cpp \
    DateTime.cpp \
    Doppler.cpp \
    Ellipsoid.cpp \
    EulerAngles.cpp \
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
    Quaternion.cpp \
    Raster.cpp \
    ResampSlc.cpp \

# products
# the library
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(PROJECT_MAJOR).$(PROJECT_MINOR).$(EXT_SO)
# the private build location
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)-$(PROJECT_MAJOR).$(PROJECT_MINOR)/lib

# what to export
EXPORT_LIBS = $(PROJ_DLL)
# the headers
EXPORT_PKG_HEADERS = \
    Attitude.h \
    Baseline.h \
    Constants.h \
    DateTime.h \
    Doppler.h \
    Ellipsoid.h \
    EulerAngles.h \
    EulerAngles.icc \
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
    Quaternion.h \
    Quaternion.icc \
    Raster.h \
    Raster.icc \
    ResampSlc.h \
    ResampSlc.icc \
    Serialization.h \
    StateVector.h \
    Tile.h \
    Tile.icc \

# build
PROJ_CXX_INCLUDES += $(EXPORT_ROOT)/include/$(PROJECT)-$(PROJECT_MAJOR).$(PROJECT_MINOR)

# standard targets
all: export

export:: $(PROJ_DLL) export-package-headers export-libraries

live:: live-headers live-package-headers live-libraries
	BLD_ACTION="live" $(MM) recurse

# configuration
# the extension of the c++ sources
EXT_CXX = cpp




# end-of-file
