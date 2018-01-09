# -*- Makefile -*-

# build a shared library
include shared/target.def

# global project settings
PROJECT = gtest_isce

# the private build location
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/lib

# the list of sources
PROJ_SRCS = \
    MinimalistPrinter.cc \

PROJ_CXX_INCLUDES = .

# products
# the library
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SO)
EXPORT_LIBS = $(PROJ_DLL)

# standard targets
all: $(PROJ_DLL) export

export:: export-libraries

# configuration
# the extension of the c++ sources
EXT_CXX = cc


# end-of-file
