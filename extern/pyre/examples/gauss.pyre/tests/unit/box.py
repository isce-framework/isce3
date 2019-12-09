#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the box shape behaves as expected
"""


def test():
    import gauss.shapes

    # instantiate
    box = gauss.shapes.box(name="box")
    box.intervals = ((0,1), (0,1))

    # check the volume
    assert box.measure() == 1.0

    # set up some interior points
    interior = [(0,0), (1,0), (1,1), (0,1), (.5, .5)]
    assert len(list(box.contains(interior))) == len(interior)

    # set up some exterior points
    exterior = [(2,0), (0,2), (-2,0), (0,-2)]
    assert len(list(box.contains(exterior))) == 0

    return box


# main
if __name__ == "__main__":
    test()


# end of file
