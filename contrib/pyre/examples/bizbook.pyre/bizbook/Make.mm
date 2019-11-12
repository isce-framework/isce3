# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = bizbook
PACKAGE = bizbook
PROJ_CLEAN = $(EXPORT_MODULEDIR)


#--------------------------------------------------------------------------
#

all: export

#--------------------------------------------------------------------------
# export

EXPORT_PYTHON_MODULES = \
    schema.py \
    __init__.py


export:: export-python-modules

# end of file
