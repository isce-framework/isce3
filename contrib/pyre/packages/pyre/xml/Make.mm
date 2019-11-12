# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = xml
# the python modules
EXPORT_PYTHON_MODULES = \
    AttributeDescriptor.py \
    DTD.py \
    Descriptor.py \
    Document.py \
    ElementDescriptor.py \
    Ignorable.py \
    Node.py \
    Reader.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
