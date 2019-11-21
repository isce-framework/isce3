# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Constraint import Constraint


# declaration
class Comparison(Constraint):
    """
    Base class for constraints that compare candidates against values
    """

    # my comparison operator and its textual representation
    tag = None
    compare = None


    # interface
    def validate(self, value, **kwds):
        """
        Check whether {value} satisfies this constraint
        """
        # if {value} compares correctly with my value
        if self.compare(value, self.value):
            # indicate success
            return value
        # otherwise, chain up
        return super().validate(value=value, **kwds)


    # meta-methods
    def __init__(self, value, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my reference value
        self.value = value
        # all done
        return


    def __str__(self):
        return "{0.tag} {0.value!r}".format(self)


# end of file
