# -*- Makefile -*-

# global project settings
PROJECT = gtest

# the private build location
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/lib

#googletest instruction for building
#g++  -shared -undefined dynamic_lookup -o ~/tools/lib/libgtest.so  -dy ./include -I./ -I../googletest/include/ -pthread -c src/gtest-all.cc

# the list of sources
PROJ_SRCS = \
    gtest.cc \
    gtest-death-test.cc \
    gtest-filepath.cc \
    gtest-port.cc \
    gtest-printers.cc \
    gtest-test-part.cc \
    gtest-typed-test.cc \

PROJ_CXX_INCLUDES = ..

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
