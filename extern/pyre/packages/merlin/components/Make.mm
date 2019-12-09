# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include merlin.def
# package name
PACKAGE = components
# the python modules
EXPORT_PYTHON_MODULES = \
    Action.py \
    Component.py \
    Curator.py \
    Dashboard.py \
    Merlin.py \
    PythonClassifier.py \
    Spell.py \
    Spellbook.py \
    exceptions.py \
    __init__.py


# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
