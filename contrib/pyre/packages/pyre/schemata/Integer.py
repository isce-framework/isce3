# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Numeric import Numeric


# declaration
class Integer(Numeric):
    """
    A type declarator for integers
    """


    # constants
    typename = 'int' # the name of my type
    complaint = 'could not coerce {0.value!r} into an integer'


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into an integer
        """
        # attempt to convert into an integer
        try:
            # for strings
            if isinstance(value, str):
                # get the interpreter to evaluate simple expressions
                value = eval(value, self.context)
            # everything must to go through the {int} constructor to get coerced correctly
            return int(value)
        # if anything whatsoever goes wrong
        except Exception as error:
            # complain
            raise self.CastingError(value=value, description=self.complaint)


    # meta-methods
    def __init__(self, default=int(), **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # all done
        return


# end of file
