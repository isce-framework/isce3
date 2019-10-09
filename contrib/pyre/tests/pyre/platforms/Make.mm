# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

#--------------------------------------------------------------------------
#

all: test

test: sanity host

sanity:
	${PYTHON} ./sanity.py

host:
	${PYTHON} ./host.py


# end of file
