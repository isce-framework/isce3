# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools


# class declaration
class Grid(list):
    """
    A logically Cartesian grid implemented as a list with a custom indexing function
    """


    # constants
    ROW_MAJOR = "row-major"
    COLUMN_MAJOR = "column-major"


    # public data
    @property
    def dimension(self):
        """
        Compute the dimension of space
        """
        # easy enough: compute the length of the zeroth item
        return len(self[0])


    @property
    def numberOfPoints(self):
        """
        Compute the number of nodes in the grid
        """
        # initialize the counter
        size = 1
        # go through the extent of each axis
        for axis in self.shape:
            # and multiply it out
            size *= axis
        # all done
        return size


    @property
    def numberOfCells(self):
        """
        Compute the number of cells in the grid
        """
        # initialize the counter
        size = 1
        # go through the extent of each axis
        for axis in self.shape:
            # and multiply it out
            size *= axis - 1
        # all done
        return size


    @property
    def points(self):
        """
        Return an iterator over the nodes in my grid
        """
        # easy enough
        yield from self


    @property
    def cells(self):
        """
        Return an iterator over the connectivity of the grid
        """
        # generate a sequence of grid indices for the corners of each cell in the grid
        for anchor in self.anchors():
            # build the connectivity of the cell anchored at this corner
            yield self.describeCellAt(anchor)
        # all done
        return


    # interface
    def boundingBox(self):
        """
        Compute the bounding box of the grid
        """
        # compute the smallest triplet of coordinates
        bmin = tuple(map(min, zip(*self)))
        # and the largest
        bmax = tuple(map(max, zip(*self)))
        # return them
        return (bmin, bmax)


    # meta-methods
    def __init__(self, shape=(), packing=ROW_MAJOR, *args, **kwds):
        # chain up
        super().__init__(*args, **kwds)
        # save my dimensions
        self.shape = tuple(shape)
        # and my packing strategy
        self.packing = packing

        # adjust my projection operator based on the requested packing
        # if the client asked for row-major
        if packing == self.ROW_MAJOR:
            # use row-major indexed access
            self.project = self.rowMajor
        # if the client asked for column-major
        elif packing == self.COLUMN_MAJOR:
            # use row-major indexed access
            self.project = self.columnMajor
        # otherwise
        else:
            # complain
            raise ValueError('unknown grid packing strategy')

        # all done
        return


    def __getitem__(self, index):
        """
        Support structured access to the cells
        """
        # attempt to
        try:
            # realize the index
            index = self.project(index)
        # if this fails
        except TypeError:
            # convert it into an integer
            index = int(index)

        # chain up
        return super().__getitem__(index)


    # implementation details
    def columnMajor(self, index):
        """
        Convert the {index} into an offset using column-major format
        """
        # to start off
        offset = 0
        product = 1
        # loop over the indices
        for i, s in zip(index, self.shape):
            # yield the current addend
            offset += i*product
            # adjust the coefficient
            product *= s
        # all done
        return offset


    def rowMajor(self, index):
        """
        Convert the {index} into an offset using row-major format
        """
        # to start off
        offset = 0
        product = 1
        # loop over the indices
        for i, s in zip(reversed(index), reversed(self.shape)):
            # yield the current addend
            offset += i*product
            # adjust the coefficient
            product *= s
        # all done
        return offset


    def verify(self):
        """
        Run some simple diagnostics
        """
        # compute the size implied by my shape
        size = 1
        for axis in self.shape: size *= axis
        # check the length
        assert len(self) == size, "wrong size: length: {}, computed size: {}".format(
            len(self), size)

        # my dimension
        dim = len(self.shape)
        # go through my contents and verify that
        for i, point in enumerate(self):
            # the space dimension of this point
            d = len(point)
            # is in the correct space
            assert d == dim, "point {}: wrong dimension: {}, not {}".format(i, d, dim)

        # all done
        return


    def anchors(self):
        """
        Compute the grid index of the corner of each cell
        """
        # form all possible tuples of grid indices, not including the upper grid boundary
        yield from itertools.product(*map(range, (axis-1 for axis in self.shape)))


    def describeCellAt(self, anchor):
        """
        Build a pair of indices for each coordinate that can be used to visit the corners of a grid
        cell in counter clockwise fashion
        """
        # get the corners
        corners = self.cornersOfCellAt(anchor)
        # project them into the grid  to get the flat index and return them
        return tuple(map(self.project, corners))


    def cornersOfCellAt(self, anchor):
        """
        Visit the corners of the cell at {anchor} in a counter clockwise fashion
        """
        # for 3d grids
        try:
            # unpack the indices
            i,j,k = anchor
        # if this fails
        except ValueError:
            # moving on
            pass
        # if it doesn't
        else:
            # build the indices of the corners of the bottom and top faces in counterclockwise
            # fashion
            return (
                (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k),
                (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1))

        # for 2d grids
        try:
            # unpack the indices
            i,j = anchor
        # if this fails
        except ValueError:
            # moving on
            pass
        # if it doesn't
        else:
            # build the indices of the corners of the bottom and top faces in counterclockwise
            # fashion
            return ((i,j), (i+1,j), (i+1,j+1), (i,j+1))

        # for the rest
        return anchor


# end of file
