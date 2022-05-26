# -*- Makefile -*-


# project meta-data
isce3.major := $(repo.major)
isce3.minor := $(repo.minor)
isce3.micro := $(repo.micro)
isce3.revision := $(repo.revision)
isce3.full := $(isce3.major).$(isce3.minor).$(isce3.micro)

# if we have cuda
ifdef cuda.dir
  isce3.cuda := 1
else
  isce3.cuda := 0
endif

# isce3 consists of python packages
isce3.packages := isce3.pkg
# libraries
isce3.libraries := isce3.lib
# python extensions
isce3.extensions := isce3.ext
# and test suites
isce3.tests := isce3.cxx.tests
# we also know how to build a number of docker images
isce3.docker-images := \
  hirsute.dev

# put it all together
isce3.assets := \
    $(isce3.packages) $(isce3.libraries) $(isce3.extensions) $(isce3.tests) \
    $(isce3.docker-images)

# external package configuration
# fftw libraries
fftw.flavor := 3 3_threads 3f 3f_threads
# cuda libraries
cuda.libraries += cufft cudart cudadevrt

# get the subprojects
include $(isce3.assets)


# end of file
