# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#

# project defaults
include pyre.def
# my subdirectories
RECURSE_DIRS = \
    toy.pyre \
    gauss.pyre

# add these if the corresponding support has been built
# postgres
ifneq ($(strip $(LIBPQ_DIR)),)

  RECURSE_DIRS += bizbook.pyre

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

# end of file
