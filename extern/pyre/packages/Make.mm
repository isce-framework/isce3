# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project global settings
include pyre.def
# my subdirectories
RECURSE_DIRS = \
    $(PACKAGES)
# the ones that are always available
PACKAGES = \
    pyre \
    journal \
    merlin \
    opal \

# the optional packages
# cuda
ifneq ($(strip $(CUDA_DIR)),)
  PACKAGES += cuda
endif

# gsl
ifneq ($(strip $(GSL_DIR)),)
  PACKAGES += gsl
endif

# mpi
ifneq ($(strip $(MPI_DIR)),)
  PACKAGES += mpi
endif

# the standard targets
all:
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

live:
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	cd $(EXPORT_ROOT); zip -r $(PYRE_ZIP) ${addprefix packages/, $(PACKAGES) --include \*.py}

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
