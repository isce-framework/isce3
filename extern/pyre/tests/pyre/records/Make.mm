# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


PROJECT = pyre

#--------------------------------------------------------------------------
#

all: test

test: sanity simple complex csv

sanity:
	${PYTHON} ./sanity.py

simple: simple-structure simple-immutable simple-mutable

simple-structure:
	${PYTHON} ./simple.py
	${PYTHON} ./simple_layout.py
	${PYTHON} ./simple_inheritance.py
	${PYTHON} ./simple_inheritance_multi.py

simple-immutable:
	${PYTHON} ./simple_immutable_data.py
	${PYTHON} ./simple_immutable_kwds.py
	${PYTHON} ./simple_immutable_conversions.py
	${PYTHON} ./simple_immutable_validations.py

simple-mutable:
	${PYTHON} ./simple_mutable_data.py
	${PYTHON} ./simple_mutable_kwds.py
	${PYTHON} ./simple_mutable_conversions.py
	${PYTHON} ./simple_mutable_validations.py

complex: complex-structure complex-immutable complex-mutable

complex-structure:
	${PYTHON} ./complex_layout.py
	${PYTHON} ./complex_inheritance.py
	${PYTHON} ./complex_inheritance_multi.py

complex-immutable:
	${PYTHON} ./complex_immutable_data.py
	${PYTHON} ./complex_immutable_kwds.py
	${PYTHON} ./complex_immutable_conversions.py
	${PYTHON} ./complex_immutable_validations.py

complex-mutable:
	${PYTHON} ./complex_mutable_data.py
	${PYTHON} ./complex_mutable_kwds.py
	${PYTHON} ./complex_mutable_conversions.py
	${PYTHON} ./complex_mutable_validations.py

csv:
	${PYTHON} ./csv_instance.py
	${PYTHON} ./csv_read_simple.py
	${PYTHON} ./csv_read_partial.py
	${PYTHON} ./csv_read_mutable.py
	${PYTHON} ./csv_read_complex.py
	${PYTHON} ./csv_bad_source.py


# end of file
