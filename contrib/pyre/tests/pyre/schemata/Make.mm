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

test: sanity types meta

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./exceptions.py

types:
	${PYTHON} ./arrays.py
	${PYTHON} ./booleans.py
	${PYTHON} ./catalogs.py
	${PYTHON} ./dates.py
	${PYTHON} ./decimals.py
	${PYTHON} ./dimensionals.py
	${PYTHON} ./floats.py
	${PYTHON} ./fractional.py
	${PYTHON} ./inets.py
	${PYTHON} ./integers.py
	${PYTHON} ./lists.py
	${PYTHON} ./mappings.py
	${PYTHON} ./paths.py
	${PYTHON} ./sets.py
	${PYTHON} ./strings.py
	${PYTHON} ./times.py
	${PYTHON} ./timestamps.py
	${PYTHON} ./tuples.py
	${PYTHON} ./uris.py

meta:
	${PYTHON} ./typed.py


# end of file
