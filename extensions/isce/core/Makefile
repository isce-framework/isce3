# Makefile for building isce::core

CC = g++
FDEBUG = -g
CFLAGS = $(FDEBUG) -Wall -std=c++11 -c -O3
# INCBASE is set using hybrid of relative and hard pathing (since we're in isce/extensions/isce/core and
# need to include isce/extensions). This will change with a full build system (ideally which will set env
# variables to handle this)
INCBASE = -I$(PWD)/../..
BUILDDIR = export

OBJS = DateTime.o Ellipsoid.o Interpolator.o LinAlg.o Orbit.o Peg.o Pegtrans.o Poly1d.o Poly2d.o Position.o

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
	ar rc libiscecore.a $(OBJS); \
	mkdir -p $(BUILDDIR); \
	mv libiscecore.a $(BUILDDIR); \
	cp *.h $(BUILDDIR); \
	rm *.o

# Individual object compilation
DateTime.o : DateTime.cpp DateTime.h
	$(CC) $(CFLAGS) $(INCBASE) DateTime.cpp

Ellipsoid.o : Ellipsoid.cpp Ellipsoid.h Constants.h LinAlg.h
	$(CC) $(CFLAGS) $(INCBASE) Ellipsoid.cpp

Interpolator.o : Interpolator.cpp Interpolator.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) Interpolator.cpp

LinAlg.o : LinAlg.cpp LinAlg.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) LinAlg.cpp

Orbit.o : Orbit.cpp Orbit.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) Orbit.cpp

Peg.o : Peg.cpp Peg.h
	$(CC) $(CFLAGS) $(INCBASE) Peg.cpp

Pegtrans.o : Pegtrans.cpp Pegtrans.h Constants.h Ellipsoid.h LinAlg.h Peg.h
	$(CC) $(CFLAGS) $(INCBASE) Pegtrans.cpp

Poly1d.o : Poly1d.cpp Poly1d.h
	$(CC) $(CFLAGS) $(INCBASE) Poly1d.cpp

Poly2d.o : Poly2d.cpp Poly2d.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) Poly2d.cpp

Position.o : Position.cpp Position.h Constants.h LinAlg.h
	$(CC) $(CFLAGS) $(INCBASE) Position.cpp

# Clean up objects, library, and build directory
clean :
	rm -f *.o; \
	rm -f libiscecore.a; \
	rm -rf $(BUILDDIR)

# Clean up all sub-directories as well
clean-all : clean
	$(MAKE) -C cuda clean
	$(MAKE) -C cython clean

