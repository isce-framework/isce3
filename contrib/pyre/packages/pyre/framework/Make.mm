# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = framework
# the python modules
EXPORT_PYTHON_MODULES = \
    Dashboard.py \
    Environ.py \
    Executive.py \
    FileServer.py \
    Linker.py \
    NameServer.py \
    Package.py \
    Priority.py \
    Pyre.py \
    Schema.py \
    Slot.py \
    SlotInfo.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
