# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = tabular
# the python modules
EXPORT_PYTHON_MODULES = \
    Chart.py \
    Column.py \
    Dimension.py \
    Inferred.py \
    Interval.py \
    Measure.py \
    Pivot.py \
    Primary.py \
    Reduction.py \
    Selector.py \
    Sheet.py \
    Surveyor.py \
    Tabulator.py \
    View.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
