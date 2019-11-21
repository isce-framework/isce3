# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
from .Tile import Tile


# declaration
class Grid:
    """
    A logically cartesian grid
    """


    # interface
    def enumerate(self):
        """
        Visit the entire grid in layout order returning ({index}, {value}) pairs
        """
        # go through the container and tile in sync
        for index, value in zip(self.tile, self.data):
            # hand the {index} and the corresponding value to the caller
            yield index, value
        # all done
        return


    # meta-methods
    def __init__(self, shape, layout=None, value=None, data=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # make a tile out of my shape and layout
        self.tile = Tile(shape=shape, layout=layout)

        # compute the tile size
        size = self.tile.size

        # if the caller provided a {data} container
        if data is not None:
            # use it
            # N.B.: it is the caller's responsibility to provide a container of the correct size
            self.data = data
        # if {value} is callable
        elif callable(value):
            # build my data by invoking it once per cell
            self.data = [ value() for _ in range(size) ]
        # otherwise
        else:
            # make a list filled with value
            self.data = [ value ] * size

        # all done
        return


    def __getitem__(self, index):
        """
        Return the value stored at {index}
        """
        # attempt to
        try:
            # cast {index} to an integer
            index = int(index)
        # if this fails
        except TypeError:
            # ask my tile do the rest
            value = self.data[self.tile.offset(index)]
        # otherwise
        else:
            # retrieve the item directly from my container
            value = self.data[index]
        # all done
        return value



    def __setitem__(self, index, value):
        """
        Return the value stored at {index}
        """
        # attempt to
        try:
            # cast {index} to an integer
            index = int(index)
        # if this fails
        except TypeError:
            # let my tile do the rest
            self.data[self.tile.offset(index)] = value
        # otherwise
        else:
            # set the item directly in my container
            self.data[index] = value
        # all done
        return


    def __len__(self):
        """
        Compute my length
        """
        # my tile knows; so does my {data} container
        return self.tile.size


# end of file
