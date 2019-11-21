# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = bizbook

RECURSE_DIRS = \
    postgres \
    sqlite \

#--------------------------------------------------------------------------
#

all: test

test::
	BLD_ACTION="test" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse


# end of file
