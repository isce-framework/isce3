# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from .. import units
# my superclass
from .Numeric import Numeric


# declaration
class Dimensional(Numeric):
    """
    A type declarator for quantities with units
    """


    # constants
    typename = 'dimensional' # the name of my type
    complaint = 'could not coerce {0.value!r} into a dimensional quantity'


    # public data
    parser = units.parser()


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a dimensional
        """
        # dimensionals go right through
        if isinstance(value, units.dimensional): return value

        # attempt to coerce strings
        try:
            # by invoking the {units} parser
            return self.parser.parse(value, context=self.context)
        # if anything whatsoever goes wring
        except Exception as error:
            # complain
            raise self.CastingError(value=value, description=self.complaint)


    def json(self, value):
        """
        Generate a JSON representation of {value}
        """
        # represent as a string
        return self.string(value)


    # meta-methods
    def __init__(self, default=units.zero, **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # all done
        return


# end of file
