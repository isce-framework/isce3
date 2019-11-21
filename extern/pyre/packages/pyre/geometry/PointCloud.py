# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class PointCloud(list):
    """
    A container of points
    """


    # types
    from .Point import Point


    # interface
    def point(self, coordinates):
        """
        Create a point at the given location
        """
        # make one
        point = self.Point(coordinates)
        # save it
        self.append(point)
        # and return it to the caller
        return point


# end of file
