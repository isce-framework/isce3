# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


PROJECT = pyre
PROJ_CLEAN += output.cfg

#--------------------------------------------------------------------------
#

all: test clean

test: sanity simple compositions

sanity:
	${PYTHON} ./sanity.py

simple: basic complex containers meta

basic:
	${PYTHON} ./booleans.py
	${PYTHON} ./decimals.py
	${PYTHON} ./floats.py
	${PYTHON} ./inets.py
	${PYTHON} ./integers.py
	${PYTHON} ./strings.py

complex:
	${PYTHON} ./dates.py
	${PYTHON} ./dimensionals.py
	${PYTHON} ./paths.py
	${PYTHON} ./times.py
	${PYTHON} ./timestamps.py
	${PYTHON} ./uris.py

containers:
	${PYTHON} ./arrays.py
	${PYTHON} ./tuples.py
	${PYTHON} ./lists.py
	${PYTHON} ./sets.py

meta:
	${PYTHON} ./istreams.py
	${PYTHON} ./ostreams.py

compositions:
	${PYTHON} ./harvesting.py
	${PYTHON} ./defaults.py
	${PYTHON} ./inheritance.py
	${PYTHON} ./filtering.py
	${PYTHON} ./converters.py

# end of file
