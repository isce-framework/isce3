# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = parsing
# the python modules
EXPORT_PYTHON_MODULES = \
    Descriptor.py \
    InputStream.py \
    Lexer.py \
    Parser.py \
    SWScanner.py \
    Scanner.py \
    Token.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
