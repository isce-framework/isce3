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
    def point(self, box):
        """
        Generate a random point in the interior of {box}
        """
        # build the point p by caling random the right number of times
        p = [ random.uniform(left, right) for left, right in box ] #@\label{line:mt:list}@
        # and return it
        return p


# end of file
