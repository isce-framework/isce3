# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
import random
import itertools
# my protocol
from .PointCloud import PointCloud

class Mersenne(pyre.component, family="gauss.meshes.mersenne", implements=PointCloud):
    """
    A point generator that uses the python builtin random number generator
    """


    # public state
    seed = pyre.properties.int(default=None)
    seed.doc = "initialization for the random number generator"


    # interface
    @pyre.export
    def points(self, count, box):
        """
        Generate {count} random points chosen from the interior of {box}
        """
        # our random number generator
        rng = self.rng.uniform
        # get starmap from itertools
        starmap = itertools.starmap
        # get the extent of the box
        intervals = box.intervals
        # loop {count} times
        while count > 0:
            # decrement the counter
            count -= 1
            # build a point by calling the random number generator as many times as there are
            # dimensions in the box specification and send it along
            yield tuple(starmap(rng, intervals))
        # all done
        return


    # meta methods
    def __init__(self, **kwds):
        super().__init__(**kwds)
        # build and seed a random number generator
        self.rng = random.Random(self.seed)
        # all done
        return


# end of file
