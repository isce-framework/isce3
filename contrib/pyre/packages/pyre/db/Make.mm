# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = db
# the python modules
EXPORT_PYTHON_MODULES = \
    Backup.py \
    Client.py \
    Collation.py \
    DataStore.py \
    FieldReference.py \
    FieldSelector.py \
    ForeignKey.py \
    Measure.py \
    Object.py \
    Persistent.py \
    Postgres.py \
    Query.py \
    Reference.py \
    SQL.py \
    SQLite.py \
    Schemer.py \
    Selector.py \
    Server.py \
    Table.py \
    actions.py \
    exceptions.py \
    expressions.py \
    literals.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
