# Makefile for building isce::core

CC = g++
# As with main core Makefile, use hybrid pathing to identify include dir
INCBASE = $(PWD)/../../../..
COREBASE = $(INCBASE)/isce/core
CFLAGS = -std=c++11 -O3
IFLAGS = -I$(INCBASE)
LFLAGS = -L$(COREBASE)/export -liscecore

.PHONY : clean tests builds

all : tests

# Build tests, running each individually, and clean up
tests : ellipsoid
	$(MAKE) clean

# Just build the unit tests but don't run or clean them
builds: ellipse-build

ellipsoid : ellipse-build
	./ellipse-test

ellipse-build : testEllipsoid.cpp $(COREBASE)/Ellipsoid.h $(COREBASE)/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testEllipsoid.cpp -o ellipse-test $(LFLAGS)

# Clean up generated test file
clean :
	rm -f ellipse-test

