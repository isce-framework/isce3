# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package
PACKAGE = config
# my subdirectories
RECURSE_DIRS = \
    cfg \
    native \
    odb \
    pfg \
    pml

# the python modules
EXPORT_PYTHON_MODULES = \
    Codec.py \
    CommandLineParser.py \
    Configurator.py \
    Loader.py \
    Shelf.py \
    events.py \
    exceptions.py \
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
