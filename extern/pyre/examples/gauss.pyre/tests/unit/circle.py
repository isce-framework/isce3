#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the gauss application package is accessible
"""


def functors():
    import gauss.functors

    one = gauss.functors.one(name="one")
    gaussian = gauss.functors.gaussian(name="gaussian")

    return one, gaussian


def integrators():
    import gauss.integrators

    montecarlo = gauss.integrators.montecarlo(name="mc")

    return montecarlo


def shapes():
    import gauss.shapes

    circle = gauss.shapes.circle(name="circle")

    return circle


# main
if __name__ == "__main__":
    functors()
    integrators()
    shapes()


# end of file
