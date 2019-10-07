# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access the package with support for rational number
import fractions
# and my superclass
from .Numeric import Numeric


class Fraction(Numeric):
    """
    A type declarator for fixed point numbers
    """


    # constants
    typename = 'fraction' # the name of my type
    ncomplaint = 'could not coerce {0.value!r) into a fraction'


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a fraction
        """
        # attempt to
        try:
            # let the constructor do its job
            return fractions.Fraction(value)
        # if anything goes wrong
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
    def __init__(self, default=fractions.Fraction(), **kwds):
        # chain up with my default
        super().__init__(default=default, **kwds)
        # all done
        return


# end of file
