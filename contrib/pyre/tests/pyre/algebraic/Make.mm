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

test: sanity structural algebra graph

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./exceptions.py

structural:
	${PYTHON} ./layout.py

algebra:
	${PYTHON} ./arithmetic.py
	${PYTHON} ./ordering.py
	${PYTHON} ./boolean.py

graph:
	${PYTHON} ./dependencies.py
	${PYTHON} ./patch.py


# end of file
