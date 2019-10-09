# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package provides access to the factories for typed properties
"""

# get the typed class
from .Property import Property as property

# for convenience, expose the typed ones
# the simple ones
bool = property.bool
complex = property.complex
decimal = property.decimal
float = property.float
inet = property.inet
int = property.int
identity = property.identity
object = property.identity
str = property.str

# the more complex types
date = property.date
dimensional = property.dimensional
path = property.path
time = property.time
uri = property.uri

# containers
array = property.array
list = property.list
set = property.set
tuple = property.tuple

# meta
istream = property.istream
ostream = property.ostream
envvar = property.envvar
envpath = property.envpath

from .Facility import Facility as facility

# meta-properties: trait descriptors for homogeneous containers; these require other trait
# descriptors to specify the type of the contents
from .Dict import Dict as dict


# the decorators
from ..descriptors import converter, normalizer, validator

# common meta-descriptors
def strings(**kwds):
    """
    A list of strings
    """
    # build a descriptor that describes a list of strings
    return list(schema=str(), **kwds)


def paths(**kwds):
    """
    A list of paths
    """
    # build a descriptor that describes a list of uris and return it
    return list(schema=path(), **kwds)


def uris(**kwds):
    """
    A list of URIs
    """
    # build a descriptor that describes a list of uris and return it
    return list(schema=uri(), **kwds)


def catalog(default={}, **kwds):
    """
    A {dict} of {list}s
    """
    # build a dictionary that maps strings to lists
    return dict(schema=list(**kwds), default=default)


# end of file
