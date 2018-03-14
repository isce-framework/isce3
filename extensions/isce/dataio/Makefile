# Makefile for building isce::dataio

CC = g++
FDEBUG = -g
CFLAGS = $(FDEBUG) -Wall -fPIC -O3 -c
GDAL_INC = #-I/path/to/gdal_priv.h/header
LIB_GDAL = -L /opt/local/lib/ -lgdal
BUILDDIR = export

OBJS = Raster.o RasterLineIter.o RasterBlockIter.o

LIBTARGET = libiscedataio
LIBTARGET_A = $(LIBTARGET).a
LIBTARGET_SO = $(LIBTARGET).so

.PHONY : clean clean-all build-cpp build-cython

all : build-cpp

build-cpp : setup $(OBJS)
	ar rc $(LIBTARGET_A) $(OBJS)
	$(CC) -shared $(LIB_GDAL) -o $(LIBTARGET_SO) $(OBJS)
	mv $(LIBTARGET_A) $(BUILDDIR)/lib
	mv $(LIBTARGET_SO) $(BUILDDIR)/lib
	#rm $(OBJS)

setup :
	mkdir -p $(BUILDDIR)/include $(BUILDDIR)/lib $(BUILDDIR)/packages

%.o : %.cpp
	$(CC) $(CFLAGS) $(GDAL_INC) $<
	cp $(<:.cpp=.h) $(BUILDDIR)/include

clean :
	rm -f $(OBJS)
	rm -f $(LIBTARGET)
	rm -rf $(BUILDDIR)

