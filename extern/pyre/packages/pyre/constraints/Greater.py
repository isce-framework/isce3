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
class Greater(Comparison):
    """
    Constraint that is satisfied when a candidate is greater than a given value
    """


    # my comparison
    compare = operator.gt
    # my tag
    tag = "greater than"


# end of file
