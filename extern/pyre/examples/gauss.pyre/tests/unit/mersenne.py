#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the MersenneTwister generator behaves as expected
"""


def test():
    import gauss

    # set the default seed
    gauss.meshes.mersenne.seed = 0
    # instantiate
    mt = gauss.meshes.mersenne(name="mt")
    # the number of points to generate
    size = 5
    # specify the box
    box = gauss.shapes.box(name="box")
    box.intervals = ((0,1), (0,1))
    # make a bunch of random points
    points = tuple(mt.points(box=box, count=size))
    # check the length
    assert len(points) == size
    # verify they all lie inside the box
    intervals = tuple(box.intervals)
    for point in points:
        for p, (left,right) in zip(point, intervals):
            assert p>=left and p<=right
    # now that we seed the random number, we can actually check we got a specific sequence of
    # points
    assert points == (
        (0.8444218515250481, 0.7579544029403025),
        (0.420571580830845, 0.25891675029296335),
        (0.5112747213686085, 0.4049341374504143),
        (0.7837985890347726, 0.30331272607892745),
        (0.4765969541523558, 0.5833820394550312)
        )

    return box


# main
if __name__ == "__main__":
    test()


# end of file
