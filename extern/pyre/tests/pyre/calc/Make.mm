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

test: sanity structural evaluators expressions interpolations memo hierarchical model

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./exceptions.py
	${PYTHON} ./node_class.py
	${PYTHON} ./node_instance.py

structural:
	${PYTHON} ./node.py
	${PYTHON} ./substitute.py
	${PYTHON} ./replace.py
	${PYTHON} ./replace_probe.py
	${PYTHON} ./patch.py

evaluators:
	${PYTHON} ./explicit.py
	${PYTHON} ./probe.py
	${PYTHON} ./containers.py
	${PYTHON} ./reference.py
	${PYTHON} ./sum.py
	${PYTHON} ./aggregators.py
	${PYTHON} ./reductors.py
	${PYTHON} ./operations.py
	${PYTHON} ./algebra.py

expressions:
	${PYTHON} ./expression.py
	${PYTHON} ./expression_escaped.py
	${PYTHON} ./expression_resolution.py
	${PYTHON} ./expression_circular.py
	${PYTHON} ./expression_syntaxerror.py
	${PYTHON} ./expression_typeerror.py

interpolations:
	${PYTHON} ./interpolation.py
	${PYTHON} ./interpolation_escaped.py
	${PYTHON} ./interpolation_circular.py

memo:
	${PYTHON} ./memo.py
	${PYTHON} ./memo_model.py
	${PYTHON} ./memo_expression.py
	${PYTHON} ./memo_interpolation.py

hierarchical:
	${PYTHON} ./hierarchical.py
	${PYTHON} ./hierarchical_patch.py
	${PYTHON} ./hierarchical_alias.py
	${PYTHON} ./hierarchical_group.py
	${PYTHON} ./hierarchical_contains.py

model:
	${PYTHON} ./model.py

# end of file
