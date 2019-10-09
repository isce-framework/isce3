# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

include pyre.def
include MPI/default.def

PROJECT = pyre
PACKAGE = mpi

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_CLEAN += $(TESTS)

MPI_ARGS = --hostfile localhost

TESTS = \
    sanity \
    world \
    group \
    group-include \
    group-exclude \
    group-setops \
    communicator \

LIBRARIES = $(EXTERNAL_LIBS)

#--------------------------------------------------------------------------

all: test clean

test: $(TESTS)
	$(MPI_EXECUTIVE) ${MPI_ARGS} -np 4 ./sanity
	$(MPI_EXECUTIVE) ${MPI_ARGS} -np 4 ./world
	$(MPI_EXECUTIVE) ${MPI_ARGS} -np 4 ./group
	$(MPI_EXECUTIVE) ${MPI_ARGS} -np 7 ./group-include
	$(MPI_EXECUTIVE) ${MPI_ARGS} -np 7 ./group-exclude
	$(MPI_EXECUTIVE) ${MPI_ARGS} -np 7 ./group-setops
	$(MPI_EXECUTIVE) ${MPI_ARGS} -np 8 ./communicator

# build
%: %.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LCXXFLAGS) $(LIBRARIES)

# end of file
