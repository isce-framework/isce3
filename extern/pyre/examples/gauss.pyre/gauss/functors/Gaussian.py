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


class Gaussian(pyre.component, family="gauss.functors.gaussian", implements=Functor):
    r"""
    Component that implements the normal distribution with mean μ and variance σ^2

        g(x; μ,σ) = \frac{1}{\sqrt{2π} σ} e^{-\frac{|x-μ|^2}{2σ^2}}

    μ and σ are implemented as component properties so that Gaussian can conform to the
    functor interface. See gauss.interfaces.functor for more details.
    """

    # public state
    mean = pyre.properties.array(default=[0])
    mean.doc = "the mean of the gaussian distribution"
    mean.aliases.add("μ")

    spread = pyre.properties.float(default=1)
    spread.doc = "the variance of the gaussian distribution"
    spread.aliases.add("σ")


    # interface
    @pyre.export
    def eval(self, points):
        """
        Compute the value of the gaussian
        """
        # access the math symbols
        from math import exp, sqrt, pi as π
        # cache the inventory items
        μ = self.μ
        σ = self.σ
        # precompute the normalization factor
        normalization = 1 / sqrt(2*π) / σ
        # and the scaling of the exposnential
        scaling = 2 * σ**2
        # loop over points and yield the computed value
        for x in points:
            # compute |x - μ|^2
            # this works as long as x and μ have the same length
            r2 = sum((x_i - μ_i)**2 for x_i, μ_i in zip(x, μ))
            # yield the value at the current x
            yield normalization * exp(- r2 / scaling)
        # all done
        return


# end of file
