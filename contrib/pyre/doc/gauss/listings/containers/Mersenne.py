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
        # create the point container using a nested list comprehension: the outer one builds
        # the container of points, the inner one builds individual points as containers of
        # random numbers within the interval of box along each coordinate axis
        sample = [
            [ random.uniform(*interval) for interval in box ]
            for i in range(n)
            ]
        # note the *interval notation in the call to uniform: it unpacks the interval and
        # supplies uniform with as many arguments as there are entities in interval

        return sample


# end of file
