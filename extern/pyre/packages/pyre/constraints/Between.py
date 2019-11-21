# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Constraint import Constraint


# declaration
class Between(Constraint):
    """
    Given {a} and {b} from a set with an ordering principle, this constraint is satisfied if
    the candidate is in {(a,b)}
    """


    # interface
    def validate(self, value, **kwds):
        """
        Check whether {candidate} satisfies this constraint
        """
        # if {candidate} is between my {low} and my {high}
        if self.low < value < self.high:
            # indicate success
            return value
        # otherwise, chain up
        return super().validate(value=value, **kwds)


    # meta-methods
    def __init__(self, low, high, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my range
        self.low = low
        self.high = high
        # all done
        return


    def __str__(self):
        return "between {0.low} and {0.high}".format(self)


# end of file
