# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

RECURSE_DIRS = \
    algebra \
    geometry \
    grid \
    memory \
    journal \
    timers \

# mpi
ifneq ($(strip $(MPI_DIR)),)
  RECURSE_DIRS += mpi
endif

# standard targets
all:
	BLD_ACTION="all" $(MM) recurse

test::
	BLD_ACTION="test" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse


# end of file
