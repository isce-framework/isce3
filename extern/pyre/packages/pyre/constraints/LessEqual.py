# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import operator
# superclass
from .Comparison import Comparison


class LessEqual(Comparison):
    """
    Constraint that is satisfied if the candidate is less than or equal to a given value
    """


    # my comparison
    compare = operator.le
    # and its textual representation
    tag = "less than or equal to"


# end of file
