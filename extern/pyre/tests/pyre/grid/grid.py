#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the package is accessible
"""


def test():
    # externals
    import itertools
    # and the package
    import pyre.grid

    # set up the tile shape
    shape = 3,4
    # the initial value of grid cells
    value = None

    # make a grid
    grid = pyre.grid.grid(shape=shape, value=value)

    # check the size of the data rep
    assert grid.tile.size == len(grid.data)
    # go through the grid
    for idx in grid.tile:
        # check that the contents are correct
        assert grid[idx] == value

    # publish
    return grid


# main
if __name__ == "__main__":
    test()
