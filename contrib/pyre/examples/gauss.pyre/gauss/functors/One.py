# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import pyre
from .Functor import Functor


class One(pyre.component, family="gauss.functors.one", implements=Functor):
    """
    The unit function
    """


    # interface
    @pyre.export
    def eval(self, points):
        """
        Compute the value of the function on the supplied points
        """
        # loop over the points and return 1 regardless
        for point in points: yield 1
        # all done
        return


# end of file
