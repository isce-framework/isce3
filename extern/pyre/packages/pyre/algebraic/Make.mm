# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package
PACKAGE = algebraic
# the python modules
EXPORT_PYTHON_MODULES = \
    AbstractNode.py \
    Algebra.py \
    Arithmetic.py \
    Boolean.py \
    Composite.py \
    Leaf.py \
    Literal.py \
    Operator.py \
    Ordering.py \
    Variable.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
