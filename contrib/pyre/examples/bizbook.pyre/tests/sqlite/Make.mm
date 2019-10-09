# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = bizbook
PROJ_CLEAN += bizbook.sql

#--------------------------------------------------------------------------
#

all: test

test: sanity create queries drop clean

sanity:
	${PYTHON} ./sanity.py

create:
	${PYTHON} ./create_tables.py
	${PYTHON} ./populate.py

queries:
	${PYTHON} ./projections.py
	${PYTHON} ./restrictions.py
	${PYTHON} ./collations.py

drop:
	${PYTHON} ./drop_tables.py


# end of file
