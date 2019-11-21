# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Surface:
    """
    A representation of a surface
    """


    # types
    from .Point import Point
    from .PointCloud import PointCloud


    # interface
    def point(self, coordinates):
        """
        Add a point to the geometry database
        """
        # pass the info along to my point cloud
        return self.points.point(coordinates)


    def triangle(self, nodes):
        """
        Add a triangle to the topology database
        """
        # easy enough
        self.triangles.append(tuple(nodes))
        # all done
        return


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # build my containers
        self.points = self.PointCloud()
        self.triangles = []
        # all done
        return


# end of file
