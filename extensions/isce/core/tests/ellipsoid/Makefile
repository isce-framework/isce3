# Makefile for building isce::core

CC = g++
INCBASE = /home/joshuac/isce_3/extensions
#INCBASE = /Users/agram/tools/isce_git/src/isce/extensions
CFLAGS = -std=c++11 -O3
IFLAGS = -I$(INCBASE)
LFLAGS = -L$(INCBASE)/isce/core/export -liscecore

.PHONY : clean tests builds

all : tests

# Build tests, running each individually, and clean up
tests : ellipsoid
	$(MAKE) clean

# Just build the unit tests but don't run or clean them
builds: ellipse-build

ellipsoid : ellipse-build
	./ellipse-test

ellipse-build : testEllipsoid.cpp $(INCBASE)/isce/core/Ellipsoid.h $(INCBASE)/isce/core/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testEllipsoid.cpp -o ellipse-test $(LFLAGS)

# Clean up generated test file
clean :
	rm -f ellipse-test

