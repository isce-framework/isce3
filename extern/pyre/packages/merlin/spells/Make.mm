# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include merlin.def
# package name
PACKAGE = spells
# the python modules
EXPORT_PYTHON_MODULES = \
    About.py \
    AssetManager.py \
    Initializer.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
