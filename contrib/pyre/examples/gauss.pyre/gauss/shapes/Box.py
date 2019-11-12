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
class Box(pyre.component, family="gauss.shapes.box", implements=Shape):
    """
    A representation of the interior of a $d$-dimensional box
    """

    # public state
    intervals = pyre.properties.array(default=((0,1),(0,1)))
    intervals.doc = "the extent of the box along each axis"


    # interface
    @pyre.export
    def measure(self):
        """
        Compute my volume
        """
        # get functools and operator
        import functools, operator
        # compute and return the volume
        return functools.reduce(
            operator.mul,
            ((right-left) for left,right in self.intervals))


    @pyre.export
    def contains(self, points):
        """
        Filter out the members of {points} that are exterior to this box
        """
        # cache my extent along each coördinate axis
        intervals = self.intervals
        # now, for each point
        for point in points:
            # for each cöordinate
            for p, (left,right) in zip(point, intervals):
                # if this point is outside the box
                if p < left or p > right:
                    # move on to the next point
                    break
            # if we got here all tests passed, so
            else:
                # this one is on the interior
                yield point
        # all done
        return


# end of file
