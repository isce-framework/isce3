# Makefile for building isce::core

CC = g++
# As with main core Makefile, use hybrid pathing to identify include dir
INCBASE = $(PWD)/../../../..
COREBASE = $(INCBASE)/isce/core
CFLAGS = -fPIC -std=c++11 -O3
IFLAGS = -I$(INCBASE)
LFLAGS = -L$(COREBASE)/export/lib -liscecore

.PHONY : clean tests builds

all : tests

# Build tests, running each individually, and clean up
tests : lut2d
	$(MAKE) clean

# Just build the unit tests but don't run or clean them
builds: lut2d-build

lut2d : lut2d-build
	./testLUT2d

lut2d-build : testLUT2d.cpp $(COREBASE)/LUT2d.h $(COREBASE)/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testLUT2d.cpp -o testLUT2d $(LFLAGS)

# Clean up generated test file
clean :
	rm -f testLUT2d
