# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re
# superclass
from .Constraint import Constraint


# declaration
class Like(Constraint):
    """
    Given a regular expression, a string satisfies this constraint if it matches the regular
    expression
    """


    # interface
    def validate(self, value, **kwds):
        """
        Check whether {value} satisfies this constraint
        """
        # if the value matches my regular expression
        if self.regexp.match(value):
            # indicate success
            return value
        # otherwise, chain up
        return super().validate(value=value, **kwds)


    # meta-methods
    def __init__(self, regexp, **kwds):
        # chain up
        super().__init__(**kwds)
        # compile my regular expression
        self.regexp = re.compile(regexp)
        # all done
        return


    def __str__(self):
        return "like {!r}".format(self.regexp.pattern)


# end of file
