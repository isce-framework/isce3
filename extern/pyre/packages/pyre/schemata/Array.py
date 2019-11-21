# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections
# superclass
from .Schema import Schema


# declaration
class Array(Schema):
    """
    The array type declarator
    """


    # constants
    typename = 'array' # the name of my type
    complaint = 'could not coerce {0.value!r} to an array'


    # interface
    def coerce(self, value, **kwds):
        """
        Convert {value} into a tuple
        """
        # evaluate the string
        if isinstance(value, str):
            # strip it
            value = value.strip()
            # if there is nothing left, return an empty tuple
            if not value: return ()
            # otherwise, ask python to process
            value = eval(value)
        # if {value} is an iterable, convert it to a tuple and return it
        if isinstance(value, collections.Iterable): return tuple(value)
        # otherwise flag it as bad input
        raise self.CastingError(value=value, description=self.complaint)


    # meta-methods
    def __init__(self, default=(), **kwds):
        # chain up, potentially with my local default value
        super().__init__(default=default, **kwds)
        # all done
        return


# end of file
