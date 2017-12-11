# Makefile for building isce::core

CC = g++
FDEBUG = -g
CFLAGS = $(FDEBUG) -Wall -fPIC -std=c++11 -O3 -c
# INCBASE is set using hybrid of relative and hard pathing (since we're in isce/extensions/isce/core
# and need to include isce/extensions). This will change with a full build system (ideally which
# will set env variables to handle this)
INCBASE = -I$(PWD)/../..
BUILDDIR = export

OBJS = Baseline.o DateTime.o Ellipsoid.o Interpolator.o LinAlg.o LUT2d.o Metadata.o Orbit.o Peg.o \
	   Pegtrans.o Poly1d.o Poly2d.o Position.o Projections.o

LIBTARGET = libiscecore.a

.PHONY : clean clean-all build-cpp build-cuda build-cython

# Build everything
all : build-cpp build-cython 
#all : build-cpp build-cython build-cuda

# Export the BUILDDIR to be visible to cuda/Makefile as MAINBUILDDIR
build-cuda : export MAINBUILDDIR=$(BUILDDIR)
# Build the CUDA objects (calls cuda/Makefile)
build-cuda : setup setup-cuda
	$(MAKE) -C cuda recurse

# Export the BUILDDIR to be visible to cython/Makefile as MAINBUILDDIR
build-cython : export MAINBUILDDIR=$(BUILDDIR)
# Build the Python module of iscecore using Cython wrappers
build-cython : setup
	$(MAKE) -C cython recurse

# Build the library and clean up
build-cpp : setup $(OBJS)
	ar rc $(LIBTARGET) $(OBJS)
	mv $(LIBTARGET) $(BUILDDIR)/lib
	cp Constants.h $(BUILDDIR)/include
	rm $(OBJS)

# Set up output directory structure
setup :
	mkdir -p $(BUILDDIR)/include $(BUILDDIR)/lib $(BUILDDIR)/packages

# Add CUDA dirs to setup if we're including
setup-cuda :
	mkdir -p $(BUILDDIR)/include/cuda $(BUILDDIR)/lib/cuda

# Individual object compilation
%.o : %.cpp
	$(CC) $(CFLAGS) $(INCBASE) $<
	cp $(<:.cpp=.h) $(BUILDDIR)/include

# Clean up objects, library, and build directory
clean :
	rm -f $(OBJS)
	rm -f $(LIBTARGET)
	rm -rf $(BUILDDIR)

# Clean up all sub-directories as well
clean-all : clean
	$(MAKE) -C cuda clean
	$(MAKE) -C cython clean

