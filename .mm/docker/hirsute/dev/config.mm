# -*- Makefile -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 1998-2021 all rights reserved


# external dependencies
# system tools
sys.prefix := /usr
sys.lib := ${sys.prefix}/lib
sys.libx86 := ${sys.lib}/x86_64-linux-gnu

# cuda
cuda.version := 11.2
cuda.dir := $(sys.prefix)
cuda.libpath := $(sys.libx86)

# eigen
eigen.version := 3.3.9
eigen.dir := $(sys.prefix)

# fftw
fftw.version := 3.3.8
fftw.dir := $(sys.prefix)

# gdal
gdal.version := 3.2.2
gdal.dir := $(sys.prefix)
gdal.incpath := $(gdal.dir)/include/gdal

# gsl
gsl.version := 2.5
gsl.dir := $(sys.prefix)

# gtest
gtest.version := 1.10
gtest.dir := $(sys.prefix)

# hdf5
hdf5.version := 1.10.6
hdf5.dir := $(sys.prefix)
hdf5.incpath := $(hdf5.dir)/include/hdf5/openmpi
hdf5.libpath := $(sys.libx86)/hdf5/openmpi

# libpq
libpq.version := 12.3
libpq.dir := $(sys.prefix)
libpq.libpath := $(sys.libx86)
libpq.incpath := $(sys.prefix)/include/postgresql

# mpi
mpi.version := 4.1.0
mpi.flavor := openmpi
mpi.dir := $(sys.prefix)/lib/x86_64-linux-gnu/openmpi
mpi.executive := mpirun

# python
python.version := 3.9
python.dir := $(sys.prefix)

# numpy
numpy.version := 1.17.4
numpy.dir := $(sys.prefix)/lib/python3/dist-packages/numpy/core

# pybind11
pybind11.version := 2.4.3
pybind11.dir = $(sys.prefix)


# local installs
usr.prefix := /usr/local
# pyre
pyre.version := 1.9.10
pyre.dir := $(usr.prefix)


# install locations
# this is necessary in order to override {mm} appending the build type to the install prefix
builder.dest.prefix := $(project.prefix)/
# install the pyton packages straight where they need to go
builder.dest.pyc := $(sys.prefix)/lib/python3/dist-packages/


# end of file
