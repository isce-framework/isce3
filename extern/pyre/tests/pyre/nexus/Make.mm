# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project defaults
include pyre.def

all: test

test: sanity nodes teams clean

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./node.py

nodes:
	${PYTHON} ./node_instantiation.py
	${PYTHON} ./node_signals.py

teams:
	${PYTHON} ./pool.py
	${PYTHON} ./pool.py --tasks=4 --team.size=2

# end of file
