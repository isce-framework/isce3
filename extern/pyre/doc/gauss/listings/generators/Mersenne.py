# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import random
from PointCloud import PointCloud

class Mersenne(PointCloud):
    """
    A point generator implemented using the Mersenne Twister random number generator that is
    available as part of the python standard library
    """

    # interface
    def points(self, n, box):
        """
        Generate {n} random points in the interior of {box}
        """
        # loop over the sample size
        for i in range(n):
            # build a point
            p = [ random.uniform(*interval) for interval in box ]
            # and make it available to the caller
            yield p #@\label{line:mt:generators:yield}@
        # all done
        return #@\label{line:mt:generators:return}@


# end of file
