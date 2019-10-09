# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .NamedTuple import NamedTuple


# declaration
class Immutable(NamedTuple):
    """
    Storage for and access to the values of immutable record instances
    """


    # private data
    __slots__ = ()


# end of file
