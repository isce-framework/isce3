# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Constraint import Constraint


# declaration
class Not(Constraint):
    """
    Constraint that is satisfied when the candidate fails to satisfy a given constraint
    """


    # interface
    def validate(self, value, **kwds):
        """
        Check whether {value} satisfies this constraint
        """
        # check
        try:
            # whether my constraint is satisfied
            self.constraint.validate(value=value, **kwds)
        # and if it is not
        except self.constraint.ConstraintViolationError:
            # indicate success
            return value
        # otherwise, chain up
        return super().validate(value=value, **kwds)


    # meta-methods
    def __init__(self, constraint, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my constraint
        self.constraint = constraint
        # all done
        return


    def __str__(self):
        return "not {0.constraint}".format(self)


# end of file
