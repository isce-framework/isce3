# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access the framework
import pyre
# my protocol
from .Shape import Shape

# declaration
class Ball(pyre.component, family="gauss.shapes.ball", implements=Shape):
    """
    A representation of the interior of a spere in $d$ dimensions
    """

    # public state
    center = pyre.properties.array(default=(0,0))
    center.doc = "the location of the center of the ball"

    radius = pyre.properties.float(default=1)
    radius.doc = "the radius of the ball"


    # interface
    @pyre.export
    def measure(self):
        """
        Compute my volume
        """
        # get functools and operator
        import functools, operator
        # get π
        from math import  pi as π
        # compute the dimension of space
        d = len(self.center)
        # for even {d}
        if d % 2 == 0:
            # for even $d$
            normalization = functools.reduce(operator.mul, range(1, d//2+1))
            # compute the volume
            return π**(d//2) * self.radius**d / normalization

        # for odd {d}
        normalization = functools.reduce(operator.mul, range(1, d+1, 2))
        return 2**((d+1)//2) * π**((d-1)//2) / normalization


    @pyre.export
    def contains(self, points):
        """
        Filter out the members of {points} that are exterior to this ball
        """
        # cache the center of the ball
        center = self.center
        # compute the radius squared
        r2 = self.radius**2
        # for each point
        for point in points:
            # compute the distance from the center
            d2 = sum((p - r)**2 for p,r in zip(point, center))
            # check whether this point is inside or outside
            if r2 >= d2:
                yield point
        # all done
        return


# end of file
