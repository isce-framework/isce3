# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Machinery for building connections to database back ends
"""


# the literals
from .literals import null, default
# cascade action markers for foreign keys
from .actions import noAction, restrict, cascade, setNull, setDefault

# orderings
from .Collation import Collation as collation
# for convenience
def ascending(fieldref):
    """
    Build a clause for the {ORDER} expression that marks {fieldref} as sorted in ascending
    order
    """
    # build and return a collation object
    return collation(fieldref=fieldref, collation="ASC")

def descending(fieldref):
    """
    Build a clause for the {ORDER} expression that marks {fieldref} as sorted in descending
    order
    """
    # build and return a collation object
    return collation(fieldref=fieldref, collation="DESC")

# primitive SQL expressions
from .expressions import (
    IsNull as isNull,
    IsNotNull as isNotNull,
    Cast as cast,
    Like as like,
    )

# field declarations
from ..records import field
# the local measure class
from .Measure import Measure as measure
# the typed measures
bool = measure.bool
date = measure.date
decimal = measure.decimal
float = measure.float
int = measure.int
str = measure.str
time = measure.time
# references to external keys
from .Reference import Reference as reference

# tables
from .Table import Table as table
# queries
from .Query import Query as query
# persistent object
from .Object import Object as object

# protocols
from .DataStore import DataStore as datastore
# components
from .SQL import SQL as sql
from .Server import Server as server
from .Client import Client as client

# supported servers
from .Backup import Backup as backup
from .SQLite import SQLite as sqlite
from .Postgres import Postgres as postgres


# templates: table rows with all fields set to None; used to update table entries
def template(table):
    """
    Build a row of {table} with all its fields set to {None}.
    """
    # build an instance, bypassing the constructor
    row = table.pyre_mutable(data=(None,)*len(table.pyre_fields))
    # and return it
    return row


# end of file
