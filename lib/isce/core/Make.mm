# -*- Makefile -*-

# global project settings
include isce.def
# package isce/core
PACKAGE = isce/core

# the list of sources
PROJ_SRCS = \
    Attitude.cpp \
    Baseline.cpp \
    BilinearInterpolator.cpp \
    BicubicInterpolator.cpp \
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
    Sinc2dInterpolator.cpp \
    Spline2dInterpolator.cpp \
    TimeDelta.cpp \

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
    Basis.h \
    Constants.h \
    DateTime.h \
    Doppler.h \
    Ellipsoid.h \
    EulerAngles.h \
    Interpolator.h \
    LUT2d.h \
    LinAlg.h \
    Matrix.h \
    Matrix.icc \
    Metadata.h \
    Orbit.h \
    Peg.h \
    Pegtrans.h \
    Pixel.h \
    Poly1d.h \
    Poly2d.h \
    Position.h \
    Projections.h \
    Quaternion.h \
    Serialization.h \
    StateVector.h \
    TimeDelta.h \
    Utilities.h \

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
