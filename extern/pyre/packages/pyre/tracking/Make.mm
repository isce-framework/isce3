# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = tracking
# the python modules
EXPORT_PYTHON_MODULES = \
    Chain.py \
    Command.py \
    File.py \
    FileRegion.py \
    NameLookup.py \
    Script.py \
    Simple.py \
    Tracker.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
