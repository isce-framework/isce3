# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Numeric import Numeric


# declaration
class Float(Numeric):
    """
    A type declarator for floats
    """


    # constants
    typename = 'float' # the name of my type
    complaint = 'could not coerce {0.value!r} into a float'


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a float
        """
        # attempt to convert into a float
        try:
            # if it is a string
            if isinstance(value, str):
                # get the interpreter to evaluate simple expressions
                value = eval(value, self.context)
            # everything has to go through the {float} constructor to get coerced correctly
            return float(value)
        # if anything whatsoever goes wrong
        except Exception as error:
            # complain
            raise self.CastingError(value=value, description=self.complaint)


    # meta-methods
    def __init__(self, default=float(), **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # all done
        return


# end of file
