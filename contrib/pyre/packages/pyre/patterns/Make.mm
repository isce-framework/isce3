# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = patterns
# the python modules
EXPORT_PYTHON_MODULES = \
    AbstractMetaclass.py \
    Accumulator.py \
    AttributeClassifier.py \
    CoFunctor.py \
    ExtentAware.py \
    Named.py \
    Observable.py \
    PathHash.py \
    Printer.py \
    Singleton.py \
    Tee.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
