# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import pyre
import gauss


class exp(pyre.component, family="gauss.functors.exp", implements=gauss.interfaces.functor):
    r"""
    A functor implementation of the form $a \exp^{\beta x}$
    """

    a = pyre.properties.float(default=1)
    β = pyre.properties.array(default=[0])


    @pyre.export
    def eval(self, points):
        """
        Evaluate my functional form over the set of {points}
        """
        # access the exponential from the math package
        from math import exp
        # cache the local values
        a = self.a
        β = self.β
        # loop over the points
        for x in points:
            # compute the exponent
            exponent = sum(x_i*β_i for x_i, β_i in zip(x, β))
            # yield the value on this point
            yield a * exp(exponent)
        # all done
        return


# end of file
