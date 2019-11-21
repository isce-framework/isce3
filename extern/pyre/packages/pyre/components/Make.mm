# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package
PACKAGE = components
# the python modules
EXPORT_PYTHON_MODULES = \
    Actor.py \
    CompatibilityReport.py \
    Component.py \
    Configurable.py \
    Foundry.py \
    Inventory.py \
    Monitor.py \
    PrivateInventory.py \
    Protocol.py \
    PublicInventory.py \
    Registrar.py \
    Requirement.py \
    Revision.py \
    Role.py \
    Tracker.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
