# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from . import gsl # the extension


# the class declaration
class RNG:
    """
    Encapsulation of the pseudo-random number generators in GSL
    """


    # constants
    available = gsl.rng_avail()

    # public data
    @property
    def algorithm(self):
        return gsl.rng_name(self.rng)

    @property
    def range(self):
        return gsl.rng_range(self.rng)


    # interface
    # basic access
    def float(self):
        """
        Return a random float in the range [0, 1)
        """
        # get one and return it
        return gsl.rng_uniform(self.rng)


    def int(self):
        """
        Return a random integer within the range of the generator
        """
        # get one and return it
        return gsl.rng_get(self.rng)


    # initialization
    def seed(self, seed=0):
        """
        Initialize the RNG with the given seed
        """
        # easy enough
        gsl.rng_set(self.rng, int(seed))
        # all done
        return self


    # meta methods
    def __init__(self, algorithm='ranlxs2', **kwds):
        super().__init__(**kwds)
        # build the RNG
        self.rng = gsl.rng_alloc(algorithm)
        # all done
        return


    # private data
    rng = None


# end of file
