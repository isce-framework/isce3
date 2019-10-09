# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = bizbook

#--------------------------------------------------------------------------
#

all: test

test: sanity create queries drop

sanity:
	${PYTHON} ./sanity.py

create:
	${PYTHON} ./create_database.py
	${PYTHON} ./create_tables.py
	${PYTHON} ./populate.py

queries:
	${PYTHON} ./projections.py
	${PYTHON} ./restrictions.py
	${PYTHON} ./collations.py

drop:
	${PYTHON} ./drop_tables.py
	${PYTHON} ./drop_database.py


# end of file
