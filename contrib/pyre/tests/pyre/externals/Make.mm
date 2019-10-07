# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

all: test

test: sanity manager configurations

sanity:
	${PYTHON} ./sanity.py

manager:
	${PYTHON} ./locate.py

configurations: blas cython gcc gsl hdf5 mpi postgres python vtk

blas:
	${PYTHON} ./blas.py
	${PYTHON} ./blas.py --blas=gslcblas
	${PYTHON} ./blas.py --blas=atlas
	${PYTHON} ./blas.py --blas=openblas
	${PYTHON} ./blas.py --blas=gslcblas#mga
	${PYTHON} ./blas.py --blas=atlas#mga
	${PYTHON} ./blas.py --blas=openblas#mga

cython:
	${PYTHON} ./cython.py
	${PYTHON} ./cython.py --cython=cython2
	${PYTHON} ./cython.py --cython=cython3
	${PYTHON} ./cython.py --cython=cython2#mga
	${PYTHON} ./cython.py --cython=cython3#mga

gcc:
	${PYTHON} ./gcc.py
	${PYTHON} ./gcc.py --gcc=gcc5
	${PYTHON} ./gcc.py --gcc=gcc5#mga

gsl:
	${PYTHON} ./gsl.py
	${PYTHON} ./gsl.py --gsl=default
	${PYTHON} ./gsl.py --gsl=default#mga

hdf5:
	${PYTHON} ./hdf5.py
	${PYTHON} ./hdf5.py --hdf5=default
	${PYTHON} ./hdf5.py --hdf5=default#mga

mpi:
	${PYTHON} ./mpi.py
	${PYTHON} ./mpi.py --mpi=mpich
	${PYTHON} ./mpi.py --mpi=openmpi
	${PYTHON} ./mpi.py --mpi=mpich#mga
	${PYTHON} ./mpi.py --mpi=openmpi#mga

postgres:
	${PYTHON} ./postgres.py
	${PYTHON} ./postgres.py --postgres=default
	${PYTHON} ./postgres.py --postgres=default#mga

python:
	${PYTHON} ./python.py
	${PYTHON} ./python.py --python=python3
	${PYTHON} ./python.py --python=python2 # this fails correctly on macports
	${PYTHON} ./python.py --python=python3#python35
	${PYTHON} ./python.py --python=python2#python27
	${PYTHON} ./python.py --python=python3#mga
	${PYTHON} ./python.py --python=python2#mga

vtk:
	${PYTHON} ./vtk.py
	${PYTHON} ./vtk.py --vtk=vtk5
	${PYTHON} ./vtk.py --vtk=vtk6
	${PYTHON} ./vtk.py --vtk=vtk6#mga


# end of file
