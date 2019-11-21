# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = gauss.pyre

#--------------------------------------------------------------------------
#

all: test

test: sanity config

sanity:
	${PYTHON} ./sanity.py

config:
	${PYTHON} ./pi.py
	${PYTHON} ./gaussian.py


# end of file
