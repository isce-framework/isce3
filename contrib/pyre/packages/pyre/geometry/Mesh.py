# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Mesh:
    """
    A simple representation of a simplicial mesh that maintains a database of points and
    connectivities
    """


    # types
    from .PointCloud import PointCloud


    # public data
    @property
    def dimension(self):
        """
        Infer the dimension of space
        """
        # easy enough
        return len(self.points[0])


    @property
    def numberOfPoints(self):
        """
        Compute the number of points in the mesh
        """
        # easy enough
        return len(self.points)


    @property
    def numberOfCells(self):
        """
        Compute the number of elements in the mesh
        """
        # easy enough
        return len(self.simplices)


    @property
    def cells(self):
        """
        Build an iterator over my simplices
        """
        # easy enough
        return self.simplices


    # interface
    def point(self, coordinates):
        """
        Create a point at the given location
        """
        # add it to my cloud and return it to the caller
        return self.points.point(coordinates=coordinates)


    def simplex(self, nodes):
        """
        Add the given simplex specification to my pile
        """
        # make one
        simplex = tuple(nodes)
        # add it to the pile
        self.simplices.append(simplex)
        # and return it to the caller
        return simplex


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # make my point database
        self.points = self.PointCloud()
        # and my connectivities
        self.simplices = []
        # all done


# end of file
