# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package
PACKAGE = calc
# the python modules
EXPORT_PYTHON_MODULES = \
    Average.py \
    Calculator.py \
    Composite.py \
    Const.py \
    Count.py \
    Datum.py \
    Dependency.py \
    Dependent.py \
    Evaluator.py \
    Expression.py \
    Filter.py \
    Hierarchical.py \
    Interpolation.py \
    Mapping.py \
    Maximum.py \
    Memo.py \
    Minimum.py \
    Node.py \
    NodeInfo.py \
    Observable.py \
    Observer.py \
    Preprocessor.py \
    Postprocessor.py \
    Probe.py \
    Product.py \
    Reactor.py \
    Reference.py \
    Sequence.py \
    Sum.py \
    SymbolTable.py \
    Unresolved.py \
    Value.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
