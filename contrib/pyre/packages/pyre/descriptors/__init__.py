# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the descriptor base class
from .Descriptor import Descriptor as stem

# decorators for value processors
from .Converter import Converter as converter
from .Normalizer import Normalizer as normalizer
from .Validator import Validator as validator


# get the schemata
from .. import schemata

# build the typed descriptors
@schemata.typed
class descriptor(stem.variable): pass

# for convenience, expose the typed ones
# the simple ones
bool = descriptor.bool
complex = descriptor.complex
decimal = descriptor.decimal
float = descriptor.float
inet = descriptor.inet
int = descriptor.int
identity = descriptor.identity
str = descriptor.str

# more complex types
date = descriptor.date
dimensional = descriptor.dimensional
path = descriptor.path
time = descriptor.time
timestamp = descriptor.timestamp
uri = descriptor.uri

# containers
array = descriptor.array
list = descriptor.list
set = descriptor.set
tuple = descriptor.tuple

# meta-types
istream = descriptor.istream
ostream = descriptor.ostream


# common meta-descriptors
def strings(default=list, **kwds):
    """
    A list of strings
    """
    # build a descriptor that describes a list of strings
    return list(schema=str(), default=default)


def uris(default=list, **kwds):
    """
    A list of URIs
    """
    # build a descriptor that describes a list of uris and return it
    return list(schema=uri(), default=default)


# end of file
