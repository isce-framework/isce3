# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access the framework
import pyre
# my protocol
from .Functor import Functor


class Constant(pyre.component, family="gauss.functors.constant", implements=Functor):
    """
    A constant function
    """

    # public state
    value = pyre.properties.float(default=1)
    value.doc = "the value of the constant functor"""

    # interface
    @pyre.export
    def eval(self, points):
        """
        Compute the value of the function on the supplied points
        """
        # local cache of the constant
        value = self.value
        # loop over the points and return my value regardless
        for point in points: yield value
        # all done
        return


# end of file
