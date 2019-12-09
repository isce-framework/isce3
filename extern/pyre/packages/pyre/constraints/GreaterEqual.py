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


# declaration
class GreaterEqual(Comparison):
    """
    Constraint that is satisfied when a candidate is greater than or equal to a given value
    """


    # my comparison
    compare = operator.ge
    # my tag
    tag = "greater than or equal to"


# end of file
