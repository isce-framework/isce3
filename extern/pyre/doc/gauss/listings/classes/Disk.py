# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from Shape import Shape

class Disk(Shape):
    """
    A representation of a circular disk
    """

    # public data
    radius = 1 # by default, a unit circle
    center = (0,0) # centered at the origin

    # interface
    def interior(self, point):
        """
        Predicate that checks whether {point} falls on my interior
        """
        r2 = self.radius**2
        x0, y0 = self.center
        x, y = point
        dx = x - x0
        dy = y - y0
        # check whether the point is exterior
        if dx*dx + dy*dy > r2:
            return False
        # otherwise, it is interior
        return True

    # meta methods
    def __init__(self, radius=radius, center=center): #@\label{line:disk:constructor}@
        self.radius = radius
        self.center = center
        return


# end of file
