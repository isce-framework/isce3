# Makefile for building just the static C++ library portion of isce::core

CC = g++
FDEBUG = -g
#FDEBUG = 
CFLAGS = $(FDEBUG) -Wall -std=c++11 -c
INCBASE = -I/home/joshuac/isce_3/extensions

OBJS = Ellipsoid.o Interpolator.o LinAlg.o Orbit.o Peg.o Pegtrans.o Poly1d.o Poly2d.o Position.o

all : lib
	mkdir -p build; mv libiscecore.a build/; cp *.h build/; rm *.o

obj : $(OBJS)

Ellipsoid.o : Ellipsoid.cpp Ellipsoid.h Constants.h LinAlg.h
	$(CC) $(CFLAGS) $(INCBASE) Ellipsoid.cpp

Interpolator.o : Interpolator.cpp Interpolator.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) Interpolator.cpp

LinAlg.o : LinAlg.cpp LinAlg.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) LinAlg.cpp

Orbit.o : Orbit.cpp Orbit.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) Orbit.cpp

Peg.o : Peg.cpp Peg.h
	$(CC) $(CFLAGS) $(INCBASE) Peg.cpp

Pegtrans.o : Pegtrans.cpp Pegtrans.h Constants.h Ellipsoid.h LinAlg.h Peg.h
	$(CC) $(CFLAGS) $(INCBASE) Pegtrans.cpp

Poly1d.o : Poly1d.cpp Poly1d.h
	$(CC) $(CFLAGS) $(INCBASE) Poly1d.cpp

Poly2d.o : Poly2d.cpp Poly2d.h Constants.h
	$(CC) $(CFLAGS) $(INCBASE) Poly2d.cpp

Position.o : Position.cpp Position.h Constants.h LinAlg.h
	$(CC) $(CFLAGS) $(INCBASE) Position.cpp

lib : $(OBJS)
	ar rc libiscecore.a $(OBJS)

.PHONY : clean

clean :
	rm -f *.o; rm -f libiscecore.a; rm -rf build/
