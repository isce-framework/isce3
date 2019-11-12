# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre

# protocols
from .Integrator import Integrator
from ..functors import functor
from ..meshes import cloud
from ..shapes import shape, box, ball


# declaration
class MonteCarlo(pyre.component, family="gauss.integrators.montecarlo", implements=Integrator):
    """
    A Monte Carlo integrator
    """

    # public state
    samples = pyre.properties.int(default=10**5)
    samples.doc = "the number of integrand evaluations"

    box = shape(default=box)
    box.doc = "the bounding box for my mesh"

    mesh = cloud()
    mesh.doc = "the generator of points at which to evaluate the integrand"

    region = shape(default=ball)
    region.doc = "the shape that defines the region of integration"

    integrand = functor()
    integrand.doc = "the functor to integrate"


    # interface
    @pyre.export
    def integrate(self):
        """
        Compute the integral as specified by my public state
        """
        # compute the normalization
        normalization = self.box.measure()/self.samples
        # get the set of points
        points = self.mesh.points(count=self.samples, box=self.box)
        # narrow the set down to the ones interior to the region of integration
        interior = self.region.contains(points)
        # sum up and scale the integrand contributions
        integral = normalization * sum(self.integrand.eval(interior))
        # and return the value
        return integral


# end of file
