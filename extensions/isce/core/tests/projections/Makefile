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
tests : geocent polar utm cea 
	$(MAKE) clean

# Just build the unit tests but don't run or clean them
builds: geocent-build

geocent : geocent-build
	./geocent-test

polar : polar-build
	./polar-test

utm: utm-build
	./utm-test

cea: cea-build
	./cea-test

geocent-build : testGeocent.cpp $(COREBASE)/Ellipsoid.h $(COREBASE)/Constants.h $(COREBASE)/Projections.h
	$(CC) $(CFLAGS) $(IFLAGS) testGeocent.cpp -o geocent-test $(LFLAGS)

polar-build: testPolar.cpp $(COREBASE)/Ellipsoid.h $(COREBASE)/Constants.h $(COREBASE)/Projections.h
	        $(CC) $(CFLAGS) $(IFLAGS) testPolar.cpp -o polar-test $(LFLAGS)

utm-build: testUTM.cpp  $(COREBASE)/Ellipsoid.h $(COREBASE)/Constants.h $(COREBASE)/Projections.h
	        $(CC) $(CFLAGS) $(IFLAGS) testUTM.cpp -o utm-test $(LFLAGS)

cea-build: testCEA.cpp  $(COREBASE)/Ellipsoid.h $(COREBASE)/Constants.h $(COREBASE)/Projections.h
	        $(CC) $(CFLAGS) $(IFLAGS) testCEA.cpp -o cea-test $(LFLAGS)

# Clean up generated test file
clean :
	rm -f geocent-test
	rm -f polar-test
	rm -f utm-test
	rm -f cea-test

