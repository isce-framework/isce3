# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package
PACKAGE = handbook
# my subdirectories
RECURSE_DIRS = \
    elements \

# the python modules
EXPORT_PYTHON_MODULES = \
    constants.py \
    __init__.py

# standard targets
all: export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-package-python-modules
	BLD_ACTION="export" $(MM) recurse

live: live-package-python-modules
	BLD_ACTION="live" $(MM) recurse

# end of file
