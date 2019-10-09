# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Triangulation:
    """
    A wrapper around a two dimensional grid that converts it into a simplicial mesh
    """


    # public data
    grid = None


    @property
    def dimension(self):
        """
        Compute the dimension of space
        """
        # just ask my grid
        return self.grid.dimension


    @property
    def numberOfPoints(self):
        """
        Compute the number of nodes in the grid
        """
        # just ask my grid
        return self.grid.numberOfPoints


    @property
    def numberOfCells(self):
        """
        Compute the number of cells in the grid
        """
        # ask my grid and multiply the result by two, since we split each quad into two triangles
        return 2*self.grid.numberOfCells


    @property
    def points(self):
        """
        Return an iterator over the nodes in my grid
        """
        # pass the request on
        return self.grid.points


    @property
    def cells(self):
        """
        Return an iterator over the connectivity of the grid
        """
        # go through the quads in my grid
        for cell in self.grid.cells:
            # the cell is already a sequence, so triangulate
            yield (cell[0], cell[1], cell[2])
            yield (cell[0], cell[2], cell[3])
        # all done
        return


    # meta-methods
    def __init__(self, grid, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my grid
        self.grid = grid
        # all done
        return


# end of file
