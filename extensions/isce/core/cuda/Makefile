# Simple Makefile to compile the various GPU-compatible objects to linkable object code.
# Deposits built objects into a new build/ directory in the source folder (will change,
# is only a temporary solution while testing things)
# Author: Joshua Cohen

NVCC = nvcc
# As in main core Makefile use hybrid pathing to identify proper include dir
CXXINCDIR = -I$(PWD)/../..
# Need to include cuda_runtime.h for gpuOrbit since it does its own memory management
# ***** ALTER BASED ON YOUR OWN DIRECTORY STRUCTURE *****
NVINCDIR = -I/usr/local/cuda/include/
# *******************************************************
BUILDDIR = export
# Point nvcc to the gcc5 bin
# ***** ALTER BASED ON YOUR OWN DIRECTORY STRUCTURE *****
NVFLAGS = -ccbin $(HOME)/opt/gcc-5.4.0/bin
# *******************************************************
# no-deprecated-gpu-targets just supporessed the warning about future deprecation of arch_sm2X,
# should be removable in CUDA 9.0 when SM_2X support is deprecated officially
FLAGS =	-std=c++11 -Wno-deprecated-gpu-targets -O3 -c

OBJS = gpuEllipsoid.o gpuLinAlg.o gpuOrbit.o gpuPeg.o gpuPegtrans.o

.PHONY : clean recurse

# Build objects and export them plus their public headers to build dir (passed in from 
# core/Makefile)
all : $(OBJS)
	mkdir -p $(BUILDDIR)
	mv $(OBJS) $(BUILDDIR)
	cp $(OBJS:.o=.h) $(BUILDDIR)

# Build objects as called from isce/core/Makefile (same as all but is moved into ../$(BUILDDIR)/cuda 
# instead of $(BUILDDIR))
recurse : $(OBJS)
	mkdir -p ../$(MAINBUILDDIR)/cuda
	mv $(OBJS) ../$(MAINBUILDDIR)/cuda
	cp $(OBJS:.o=.h) ../$(MAINBUILDDIR)/cuda

%.o : %.cu
	$(NVCC) $(NVFLAGS) $(FLAGS) $(CXXINCDIR) $(NVINCDIR) $<

# Clean objects and build/ dir
clean : 
	rm -f $(OBJS)
	rm -rf $(BUILDDIR)

