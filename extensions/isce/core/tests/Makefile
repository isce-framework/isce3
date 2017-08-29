# Makefile for building isce::core

CC = g++
INCBASE = /home/joshuac/isce_3/extensions
CFLAGS = -std=c++11 -O3
IFLAGS = -I$(INCBASE)
LFLAGS = -L$(INCBASE)/isce/core/export -liscecore

.PHONY : clean 

# Build tests
all : tests
	./test; \
	$(MAKE) clean

tests : orbit-test

orbit-test : testOrbit.cpp $(INCBASE)/isce/core/Orbit.h $(INCBASE)/isce/core/Constants.h
	$(CC) $(CFLAGS) $(IFLAGS) testOrbit.cpp -o test $(LFLAGS)

# Clean up generated test file
clean :
	rm -f test

