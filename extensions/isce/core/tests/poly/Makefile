# Makefile for building isce::core

CC = g++
# As with main core Makefile, use hybrid pathing to identify include dir
INCBASE = $(PWD)/../../../..
COREBASE = $(INCBASE)/isce/core
CFLAGS = -std=c++11 -O3
IFLAGS = -I$(INCBASE)
#LFLAGS = -L$(COREBASE)/export/lib -liscecore
LFLAGS = -L$(COREBASE)/export/lib -liscecore

.PHONY : clean tests builds

all : tests

# Build tests, running each individually, and clean up
tests : poly1d
	$(MAKE) clean

# Just build the unit tests but don't run or clean them
builds: poly1d-build poly2d-build

poly1d : poly1d-build
	./poly1d-test; \
    rm -f poly1d-test

poly2d : polar-build
	./poly2d-test; \
    rm -f poly2d-test

poly1d-build : testPoly1d.cpp $(COREBASE)/Poly1d.h
	$(CC) $(CFLAGS) $(IFLAGS) testPoly1d.cpp -o poly1d-test $(LFLAGS)

poly2d-build: testPoly2d.cpp $(COREBASE)/Poly2d.h
	        $(CC) $(CFLAGS) $(IFLAGS) testPoly2d.cpp -o poly2d-test $(LFLAGS)

# Clean up generated test file
clean :
	rm -f poly1d-test
	rm -f poly2d-test
