# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Schema import Schema


# declaration
class String(Schema):
    """
    A type declarator for strings
    """


    # constants
    typename = 'str' # the name of my type
    complaint = 'could not coerce {0.value!r} into a string'


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a string
        """
        # attempt to
        try:
            # let the constructor do its job
            return str(value)
        # if anything goes wrong
        except Exception as error:
            # complain
            raise self.CastingError(value=value, description=self.complaint)


    # meta-methods
    def __init__(self, default=str(), **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # all done
        return


# end of file
