# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import functools, operator


# declaration
class Tile:
    """
    Encapsulation of the shape and layout of the contents of a grid that enable the separation
    of the topological aspects of a grid from its memory layout

    Tiles are defined by providing two pieces of information

      - {shape}: a tuple of the extent of each grid index
      - {layout}: the packing order of the indices
    """

    # public data
    shape = ()
    layout = ()
    size = 0


    # interface
    def offset(self, index):
        """
        Compute the offset of the cell at the given {index} value
        """
        # check whether
        try:
            # the index is already an integer
            index = int(index)
        # if it is not
        except TypeError:
            # no worries
            pass
        # if it is
        else:
            # nothing further to do; bounds check maybe?
            return index

        # get my shape
        shape = self.shape
        # initialize the offset
        offset = 0
        # and the running product
        product = 1
        # go through the axes in layout order
        for axis in self.layout:
            # get the index value
            value = index[axis]
            # and the limit from the shape
            limit = shape[axis]
            # check consistency
            assert value < limit, "index {} out range".format(index)
            # update the offset
            offset += value * product
            # update the product
            product *= limit
        # all done
        return offset


    def index(self, offset):
        """
        Compute the index that corresponds to the given {offset}
        """
        # ensure the {offset} is an integer
        offset = int(offset)
        # unpack my shape and my layout
        shape = self.shape
        layout = self.layout
        # initialize the index
        index = [0] * len(shape)
        # and the running product
        product = self.size

        # check consistency
        assert offset < product, "offset {} out of range".format(offset)

        # loop in reverse packing order
        for axis in reversed(layout):
            # pull the current shape limit out of the product
            product //= shape[axis]
            # compute the index
            index[axis] = offset // product
            # adjust the offset
            offset %= product

        # freeze and return
        return tuple(index)


    def visit(self, begin, end, layout):
        """
        Generate a sequence of indices in the range {begin} to {end} in {layout} order
        """
        # initialize my current value
        current = list(begin)

        # for ever
        while True:
            # make the current value available
            yield tuple(current)
            # now, pull axes in layout order and attempt to increment each one until one
            # doesn't overflow; on overflow, reset the value in {current} back to the start
            # point and try the next axis
            for axis in layout:
                # get the associated limit
                limit = end[axis]
                # get the current value for this axis and increment it by one
                value = current[axis] + 1
                # if it didn't overflow
                if value < limit:
                    # store it
                    current[axis] = value
                    # and stop looking any further
                    break
                # otherwise, we overflowed; set this axis to its starting value and grab the
                # next one
                current[axis] = begin[axis]
            # if the loop terminates naturally
            else:
                # every axis has overflowed, so we are done
                break

        # all done
        return


    # meta-methods
    def __init__(self, shape, layout=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # realize and freeze the shape
        self.shape = shape = tuple(shape)
        # and the layout
        self.layout = layout = tuple(layout or reversed(range(len(shape))))
        # compute the capacity of this tile
        self.size = functools.reduce(operator.mul, shape, 1)

        # consistency checks
        # verify that the layout consists of unique values
        assert len(set(layout)) == len(layout)
        # that the smallest one is 0
        assert min(layout) == 0
        # and that the largest one is one less than the length
        assert max(layout) == len(layout) - 1
        # verify that every index is represented in the layout
        assert len(shape) == len(layout)

        # all done
        return


    def __getitem__(self, index):
        # return the offset that corresponds to the given index
        return self.offset(index)


    def __iter__(self):
        # visit the entire tile in layout order
        yield from self.visit(begin=(0,)*len(self.shape), end=self.shape, layout=self.layout)
        # all done
        return


# end of file
