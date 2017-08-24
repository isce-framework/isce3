# Simple Makefile to compile the various GPU-compatible objects to linkable object code.
# Deposits built objects into a new build/ directory in the source folder (will change,
# is only a temporary solution while testing things)
# Author: Joshua Cohen

NVCC = nvcc
# Alter based on own directory structure
CXXINCDIR = -I/home/joshuac/isce_3/extensions
# Need to include cuda_runtime.h for gpuOrbit since it does its own memory management
# MIGHT NEED TO ALTER BASED ON YOUR OWN SYSTEM LAYOUT
NVINCDIR = -I/usr/local/cuda/include/
# no-deprecated-gpu-targets just supporessed the warning about future deprecation of arch_sm2X
FLAGS =	-std=c++11 -Wno-deprecated-gpu-targets -c

OBJS = gpuEllipsoid.o gpuLinAlg.o gpuOrbit.o gpuPeg.o gpuPegtrans.o

all : objbuild
	mkdir -p build; mv *.o build/; cp -f *.h build/

objbuild : $(OBJS)

gpuEllipsoid.o : gpuEllipsoid.cu gpuEllipsoid.h gpuLinAlg.h
	$(NVCC) $(FLAGS) $(CXXINCDIR) gpuEllipsoid.cu

gpuLinAlg.o : gpuLinAlg.cu gpuLinAlg.h
	$(NVCC) $(FLAGS) $(CXXINCDIR) gpuLinAlg.cu

gpuOrbit.o : gpuOrbit.cu gpuOrbit.h
	$(NVCC) $(FLAGS) $(CXXINCDIR) $(NVINCDIR) gpuOrbit.cu

gpuPeg.o : gpuPeg.cu gpuPeg.h
	$(NVCC) $(FLAGS) $(CXXINCDIR) gpuPeg.cu

gpuPegtrans.o : gpuPegtrans.cu gpuPegtrans.h gpuEllipsoid.h gpuLinAlg.h gpuPeg.h
	$(NVCC) $(FLAGS) $(CXXINCDIR) gpuPegtrans.cu

.PHONY : clean

clean : 
	rm -f *.o; rm -rf build/
