# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = config/pml
# the python modules
EXPORT_PYTHON_MODULES = \
    Bind.py \
    Component.py \
    Configuration.py \
    Document.py \
    Node.py \
    PML.py \
    Package.py \
    __init__.py


# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
