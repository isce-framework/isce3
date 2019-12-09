# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include merlin.def
# package name
PACKAGE = assets
# the python modules
EXPORT_PYTHON_MODULES = \
    Asset.py \
    AssetContainer.py \
    PythonModule.py \
    PythonPackage.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
