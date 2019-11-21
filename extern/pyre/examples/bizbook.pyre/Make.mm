# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


PROJECT = bizbook.pyre

RECURSE_DIRS = \
    bizbook \
    tests

#--------------------------------------------------------------------------
#

all:
	BLD_ACTION="all" $(MM) recurse
	BLD_ACTION="clean" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse


# end of file
