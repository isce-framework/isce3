# Makefile for building isce::core

CC = g++
FDEBUG = -g
CFLAGS = $(FDEBUG) -Wall -std=c++11 -c -O3
# INCBASE is set using hybrid of relative and hard pathing (since we're in isce/extensions/isce/core and
# need to include isce/extensions). This will change with a full build system (ideally which will set env
# variables to handle this)
INCBASE = -I$(PWD)/../..
BUILDDIR = export

OBJS = DateTime.o Ellipsoid.o Interpolator.o LinAlg.o Orbit.o Peg.o Pegtrans.o Poly1d.o Poly2d.o Position.o Projections.o

LIBTARGET = libiscecore.a

.PHONY : clean clean-all build-cpp build-cuda build-cython

# Build everything
all : build-cpp build-cython 
#all : build-cpp build-cython build-cuda

# Export the BUILDDIR to be visible to cuda/Makefile as MAINBUILDDIR
build-cuda : export MAINBUILDDIR=$(BUILDDIR)
# Build the CUDA objects (calls cuda/Makefile)
build-cuda :
	$(MAKE) -C cuda recurse

# Export the BUILDDIR to be visible to cython/Makefile as MAINBUILDDIR
build-cython : export MAINBUILDDIR=$(BUILDDIR)
# Build the Python module of iscecore using Cython wrappers
build-cython :
	$(MAKE) -C cython recurse

# Build the library and clean up
build-cpp : $(OBJS)
	ar rc $(LIBTARGET) $(OBJS)
	mkdir -p $(BUILDDIR)
	mv $(LIBTARGET) $(BUILDDIR)
	cp $(OBJS:.o=.h) $(BUILDDIR)
	rm $(OBJS)

# Individual object compilation
%.o : %.cpp
	$(CC) $(CFLAGS) $(INCBASE) $<

# Clean up objects, library, and build directory
clean :
	rm -f $(OBJS)
	rm -f $(LIBTARGET)
	rm -rf $(BUILDDIR)

# Clean up all sub-directories as well
clean-all : clean
	$(MAKE) -C cuda clean
	$(MAKE) -C cython clean

