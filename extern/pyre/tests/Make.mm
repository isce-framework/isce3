# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

RECURSE_DIRS = \
    python \
    libpyre \
    pyre \
    journal \
    merlin \
    sqlite \
    opal \

# the optional packages
# cuda
ifneq ($(strip $(CUDA_DIR)),)
  RECURSE_DIRS += cuda
endif

# mpi
ifneq ($(strip $(MPI_DIR)),)
  RECURSE_DIRS += mpi
endif

# gsl
ifneq ($(strip $(GSL_DIR)),)
  RECURSE_DIRS += gsl
endif

# postgres
ifneq ($(strip $(LIBPQ_DIR)),)
  RECURSE_DIRS += postgres
endif

# standard targets
all:
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

live:

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
