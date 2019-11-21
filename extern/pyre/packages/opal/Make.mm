# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include opal.def
# package name
PACKAGE = opal
# my sibfolders
RECURSE_DIRS = \
    html \
    shells \
# the python modules
EXPORT_PYTHON_MODULES = \
    __init__.py

# standard targets
all: export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-python-modules
	BLD_ACTION="export" $(MM) recurse

live: live-python-modules
	BLD_ACTION="live" $(MM) recurse

# end of file
