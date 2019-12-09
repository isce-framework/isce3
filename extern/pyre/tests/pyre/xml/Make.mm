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

test: sanity documents

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./exceptions.py
	${PYTHON} ./reader.py
	${PYTHON} ./document.py

documents:
	${PYTHON} ./blank.py
	${PYTHON} ./empty.py
	${PYTHON} ./namespaces.py
	${PYTHON} ./schema.py
	${PYTHON} ./fs.py
	${PYTHON} ./fs_namespaces.py
	${PYTHON} ./fs_schema.py
	${PYTHON} ./fs_extra.py


# end of file
