# Makefile for building isce::core

CC = g++
INCBASE = /home/joshuac/isce_3/extensions
CFLAGS = -std=c++11 -O3
IFLAGS = -I$(INCBASE)
LFLAGS = -L$(INCBASE)/isce/core/export -liscecore

.PHONY : clean tests builds

all : tests

# Build tests, running each individually, and clean up
tests : orbit
	$(MAKE) clean

# Just build the unit tests but don't run or clean them
builds : orbit-build

orbit : orbit-build
	./orbit-test

orbit-build : testOrbit.cpp $(INCBASE)/isce/core/Orbit.h $(INCBASE)/isce/core/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testOrbit.cpp -o orbit-test $(LFLAGS)

# Clean up generated test file
clean :
	rm -f orbit-test

