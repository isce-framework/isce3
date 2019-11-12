# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre

# declaration
class Integrator(pyre.protocol, family="gauss.integrators"):
    """
    Protocol declarator for integrators
    """

    # access to the required protocols
    from ..shapes.Shape import Shape as shape
    from ..functors.Functor import Functor as functor

    # public state
    region = shape()
    region.doc = "the region of integration"

    integrand = functor()
    integrand.doc = "the functor to integrate"

    # my preferred implementation
    @classmethod
    def pyre_default(cls, **kwds):
        # use {MonteCarlo} by default
        from .MonteCarlo import MonteCarlo
        return MonteCarlo

    # interface
    @pyre.provides
    def integrate(self):
        """
        Compute the integral of {integrand} over {region}
        """


# end of file
