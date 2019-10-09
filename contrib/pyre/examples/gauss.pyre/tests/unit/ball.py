#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the ball shape behaves as expected
"""


def test():
    import gauss
    from math import pi as π

    # instantiate
    ball = gauss.shapes.ball(name="ball")
    ball.radius = 1.0
    ball.center = [0.0, 0.0]

    # check the volume
    assert ball.measure() == π

    # set up some interior points
    interior = [(0,0), (1,0), (0,1), (-1,0), (0,-1)]
    assert len(list(ball.contains(interior))) == len(interior)

    # set up some exterior points
    exterior = [(2,0), (0,2), (-2,0), (0,-2)]
    assert len(list(ball.contains(exterior))) == 0

    return ball


# main
if __name__ == "__main__":
    test()


# end of file
