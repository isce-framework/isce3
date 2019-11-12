# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project defaults
include pyre.def


all: test

test: sanity tile grid

sanity:
	${PYTHON} ./sanity.py

tile:
	${PYTHON} ./tile.py

grid:
	${PYTHON} ./grid.py


# end of file
