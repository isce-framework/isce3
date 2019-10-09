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

test: sanity tables queries persistence

sanity:
	${PYTHON} ./sanity.py

tables:
	${PYTHON} ./table_declaration.py
	${PYTHON} ./table_inheritance.py
	${PYTHON} ./table_create.py
	${PYTHON} ./table_references.py
	${PYTHON} ./table_annotations.py
	${PYTHON} ./table_delete.py
	${PYTHON} ./table_instantiation.py
	${PYTHON} ./table_insert.py
	${PYTHON} ./table_update.py

queries:
	${PYTHON} ./query_star.py
	${PYTHON} ./query_projection.py
	${PYTHON} ./query_projection_expressions.py
	${PYTHON} ./query_projection_multitable.py
	${PYTHON} ./query_restriction.py
	${PYTHON} ./query_collation.py
	${PYTHON} ./query_collation_explicit.py
	${PYTHON} ./query_collation_expression.py
	${PYTHON} ./query_inheritance.py

persistence:
	${PYTHON} ./persistent_declaration.py


# end of file
