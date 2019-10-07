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
    # get the package
    import pyre.grid
    # set up the tile shape
    shape = 2,3,4,5
    # for every possible layout order
    for layout in itertools.permutations(range(len(shape))):
        # make a tile
        t = pyre.grid.tile(shape, layout)
        # iterate in packing order
        for offset, index in enumerate(t):
            # compute the offset that corresponds to this index and compare with the expected value
            assert offset == t.offset(index)
            # and the index that corresponds to this offset
            assert index == t.index(offset)

    return


# main
if __name__ == "__main__":
    test()
