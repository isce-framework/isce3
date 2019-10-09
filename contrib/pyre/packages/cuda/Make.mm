# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include cuda.def
# package name
PACKAGE = cuda
# the python modules
EXPORT_PYTHON_MODULES = \
    Device.py \
    DeviceManager.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-python-modules

live: live-python-modules

# end of file
