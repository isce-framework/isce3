# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

PROJ_CLEAN += shells.log

#--------------------------------------------------------------------------
#

all: test

test: sanity color launching clean

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./application_sanity.py
	${PYTHON} ./application_instantiation.py
	${PYTHON} ./application_inheritance.py
	${PYTHON} ./application_namespace.py
	${PYTHON} ./script_sanity.py
	${PYTHON} ./script_instantiation.py
	${PYTHON} ./fork_sanity.py
	${PYTHON} ./fork_instantiation.py
	${PYTHON} ./daemon_sanity.py
	${PYTHON} ./daemon_instantiation.py

color:
	${PYTHON} ./colors256.py
	${PYTHON} ./colors24bit.py

launching:
	${PYTHON} ./script_launching.py
	${PYTHON} ./fork_launching.py
	${PYTHON} ./daemon_launching.py


# end of file
