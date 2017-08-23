# Simple Makefile to compile the various GPU-compatible objects to linkable object code
# Author: Joshua Cohen

NVCC = nvcc
CXXINCDIR = -I../include/
# Need to include cuda_runtime.h for gpuOrbit since it does its own memory management
NVINCDIR = -I/usr/local/cuda/include/
FLAGS =	-std=c++11 -c -Wno-deprecated-gpu-targets

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
