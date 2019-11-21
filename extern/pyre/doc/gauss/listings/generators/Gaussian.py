# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from Functor import Functor

class Gaussian(Functor):
    """
    An implementation of the normal distribution with mean #@$\mu$@ and variance #@$\sigma^2$@
    """

    # public data
    mean = 0
    spread = 1

    # interface
    def eval(self, points):
        """
        Compute the value of the gaussian
        """
        # access the math symbols
        from math import exp, sqrt, pi
        # cache the shape information
        mean = self.mean
        spread = self.spread
        # precompute the normalization factor and the exponent scaling
        normalization = 1 / sqrt(2*pi) / spread
        scaling = 2 * spread**2
        # loop over points and yield the computed value
        for p in points:
            # compute the norm |p - mean|^2
            # this works as long as p and mean have the same length
            r2 = sum((p_i - mean_i)**2 for p_i, mean_i in zip(p, mean))
            # yield the value at the current p
            yield normalization * exp(- r2/scaling)
        # all done
        return

    # meta methods
    def __init__(self, mean=mean, spread=spread):
        self.mean = mean
        self.spread = spread
        return


# end of file
