# Makefile for building isce::dataio

CC = g++
FDEBUG = -g
CFLAGS = $(FDEBUG) -Wall -fPIC -O3 -c
GDAL_INC = #-I/path/to/gdal_priv.h/header
BUILDDIR = export

OBJS = Raster.o RasterLineIter.o RasterBlockIter.o

LIBTARGET = libiscedataio.a

.PHONY : clean clean-all build-cpp build-cython

all : build-cpp

build-cpp : setup $(OBJS)
	ar rc $(LIBTARGET) $(OBJS)
	mv $(LIBTARGET) $(BUILDDIR)/lib
	rm $(OBJS)

setup :
	mkdir -p $(BUILDDIR)/include $(BUILDDIR)/lib $(BUILDDIR)/packages

%.o : %.cpp
	$(CC) $(CFLAGS) $(GDAL_INC) $<
	cp $(<:.cpp=.h) $(BUILDDIR)/include

clean :
	rm -f $(OBJS)
	rm -f $(LIBTARGET)
	rm -rf $(BUILDDIR)

