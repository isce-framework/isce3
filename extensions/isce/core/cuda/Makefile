# Simple Makefile to compile the various GPU-compatible objects to linkable object code.
# Deposits built objects into a new build/ directory in the source folder (will change,
# is only a temporary solution while testing things)
# Author: Joshua Cohen

NVCC = nvcc
# Alter based on own directory structure
CXXINCDIR = -I/home/joshuac/isce_3/extensions
# Need to include cuda_runtime.h for gpuOrbit since it does its own memory management
# Alter based on own directory structure
NVINCDIR = -I/usr/local/cuda/include/
BUILDDIR = export
# no-deprecated-gpu-targets just supporessed the warning about future deprecation of arch_sm2X
FLAGS =	-std=c++11 -Wno-deprecated-gpu-targets -c

OBJS = gpuEllipsoid.o gpuLinAlg.o gpuOrbit.o gpuPeg.o gpuPegtrans.o

.PHONY : clean obj sub-make

# Build objects and export them plus their public headers to build dir (passed in from core/Makefile)
all : obj
	mkdir -p $(BUILDDIR); \
	mv *.o $(BUILDDIR); \
	cp *.h $(BUILDDIR)

# Build objects as called from isce/core/Makefile (same as all but is moved into ../$(BUILDDIR)/cuda instead of $(BUILDDIR))
sub-make : obj
	mkdir -p ../$(MAINBUILDDIR)/cuda; \
	mv *.o ../$(MAINBUILDDIR)/cuda; \
	cp *.h ../$(MAINBUILDDIR)/cuda

# Just build the objects in-place
obj : $(OBJS)

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

# Clean objects and build/ dir
clean : 
	rm -f *.o; \
	rm -rf $(BUILDDIR)

