# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = records
# the python modules
EXPORT_PYTHON_MODULES = \
    Accessor.py \
    CSV.py \
    Calculator.py \
    Compiler.py \
    Evaluator.py \
    Extractor.py \
    Immutable.py \
    Mutable.py \
    NamedTuple.py \
    Record.py \
    Selector.py \
    Templater.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
