# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = constraints
# the python modules
EXPORT_PYTHON_MODULES = \
    And.py \
    Between.py \
    Comparison.py \
    Constraint.py \
    Equal.py \
    Greater.py \
    GreaterEqual.py \
    Less.py \
    LessEqual.py \
    Like.py \
    Not.py \
    Or.py \
    Set.py \
    Subset.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
