# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Constraint import Constraint


# declaration
class Subset(Constraint):
    """
    Constraint that is satisfied when the candidate is a subset of a given set
    """


    # interface
    def validate(self, value, **kwds):
        """
        Check whether {value} satisfies this constraint
        """
        # if my set is a superset of {value}
        if self.choices.issuperset(value):
            # indicate success
            return value
        # otherwise, chain up
        return super().validate(value=value, **kwds)


    # meta-methods
    def __init__(self, choices, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my choices
        self.choices = set(choices)
        # all done
        return


    def __str__(self):
        return "a subset of {!r}".format(self.choices)


# end of file
