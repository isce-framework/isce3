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

test:
	${PYTHON} ./format.py
	${PYTHON} ./moditer.py
	${PYTHON} ./files.py
	${PYTHON} ./classattr.py
	${PYTHON} ./slots.py
	${PYTHON} ./decorators.py
	${PYTHON} ./descriptors.py
	${PYTHON} ./dict_in.py
	${PYTHON} ./dict_update.py
	${PYTHON} ./functions.py
	${PYTHON} ./initialization.py
	${PYTHON} ./locale_codec.py
	${PYTHON} ./inheritance_shadow.py
	${PYTHON} ./inheritance_multiple.py
	${PYTHON} ./inheritance_properties.py
	${PYTHON} ./metaclass.py
	${PYTHON} ./metaclass_interface.py
	${PYTHON} ./metaclass_callsequence.py
	${PYTHON} ./metaclass_dict.py
	${PYTHON} ./metaclass_prime.py
	${PYTHON} ./metaclass_kwds.py
	${PYTHON} ./metaclass_dynamic.py
	${PYTHON} ./metaclass_inheritance.py
	${PYTHON} ./metaclass_refcount.py
	${PYTHON} ./expressions.py
	${PYTHON} ./arithmetic.py
	${PYTHON} ./algebra.py
	${PYTHON} ./execv.py

# end of file
