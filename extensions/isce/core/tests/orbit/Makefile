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
tests : orbit
	$(MAKE) clean

# Just build the unit tests but don't run or clean them
builds : orbit-build orbit-bounds orbit-sort

orbit : orbit-build orbit-bounds orbit-sort
	./orbit-test
	./orbit-bounds
	./orbit-sort

orbit-build : testOrbit.cpp $(COREBASE)/Orbit.h $(COREBASE)/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testOrbit.cpp -o orbit-test $(LFLAGS)

orbit-bounds: testOrbitBounds.cpp $(COREBASE)/Orbit.h $(COREBASE)/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testOrbitBounds.cpp -o orbit-bounds $(LFLAGS)

orbit-sort: testOrbitSort.cpp $(COREBASE)/Orbit.h $(COREBASE)/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testOrbitSort.cpp -o orbit-sort $(LFLAGS)

# Clean up generated test file
clean :
	rm -f orbit-test
	rm -f orbit-bounds
	rm -f orbit-sort

