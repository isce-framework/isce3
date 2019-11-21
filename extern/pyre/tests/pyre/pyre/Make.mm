# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre
PROJ_TIDY += __pycache__

# the standard targets
all: test clean

test: sanity api traits regressions

sanity:
	${PYTHON} ./sanity.py

api:
	${PYTHON} ./loadConfiguration.py
	${PYTHON} ./resolve.py

traits:
	${PYTHON} ./spaces.py

regressions:
	${PYTHON} ./defaults.py
	${PYTHON} ./play.py

# end of file
