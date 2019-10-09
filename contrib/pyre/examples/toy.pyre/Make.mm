# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project settings
include pyre.def

# add these to the clean pile
PROJ_TIDY += __pycache__

all: test clean

test:
	${PYTHON} ./run.py

# end of file
