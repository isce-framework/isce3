# Makefile for building the static library (separately from scons)

OBJS = Ellipsoid.o Interpolator.o LinAlg.o Orbit.o Peg.o Pegtrans.o Poly1d.o Poly2d.o Position.o
INCL = -Iinclude/
CC = g++
DEBUG = -g
CFLAGS = -c -Wall -std=c++11 $(DEBUG)
LFLAGS = -Wall $(DEBUG)

all : libIsce.a

obj : $(OBJS)

Ellipsoid.o : src/Ellipsoid.cpp include/Ellipsoid.h LinAlg.o
	$(CC) $(CFLAGS) $(INCL) src/Ellipsoid.cpp

Interpolator.o : src/Interpolator.cpp include/Interpolator.h
	$(CC) $(CFLAGS) $(INCL) src/Interpolator.cpp

LinAlg.o : src/LinAlg.cpp include/LinAlg.h
	$(CC) $(CFLAGS) $(INCL) src/LinAlg.cpp

Orbit.o : src/Orbit.cpp include/Orbit.h
	$(CC) $(CFLAGS) $(INCL) src/Orbit.cpp

Peg.o : src/Peg.cpp include/Peg.h
	$(CC) $(CFLAGS) $(INCL) src/Peg.cpp

Pegtrans.o : src/Pegtrans.cpp include/Pegtrans.h Ellipsoid.o LinAlg.o Peg.o
	$(CC) $(CFLAGS) $(INCL) src/Pegtrans.cpp

Poly1d.o : src/Poly1d.cpp include/Poly1d.h
	$(CC) $(CFLAGS) $(INCL) src/Poly1d.cpp

Poly2d.o : src/Poly2d.cpp include/Poly2d.h
	$(CC) $(CFLAGS) $(INCL) src/Poly2d.cpp

Position.o : src/Position.cpp include/Position.h LinAlg.o
	$(CC) $(CFLAGS) $(INCL) src/Position.cpp

libIsce.a : $(OBJS)
	ar rc libIsce.a $(OBJS)

.PHONY : clean
clean :
	rm *.o
