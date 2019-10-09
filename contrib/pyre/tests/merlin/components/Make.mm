# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

PROJ_CLEAN += \
    .merlin/project.pickle

#--------------------------------------------------------------------------
#

all: test

test: sanity merlin clean

sanity:
	${PYTHON} ./sanity.py

merlin:
	${PYTHON} ./merlin_shell.py
	${PYTHON} ./merlin_spell.py
	${PYTHON} ./merlin_curator.py
	${PYTHON} ./merlin_packages.py


# end of file
