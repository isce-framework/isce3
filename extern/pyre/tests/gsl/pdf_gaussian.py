#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the gaussian pdf
"""


def test():
    # math
    from math import pi, sqrt, exp
    # access the package
    import gsl

    # the σ of the distribution
    σ = 2
    # build a random number generator
    rng = gsl.rng()
    # build a gaussian distribution
    gaussian = gsl.pdf.gaussian(mean=0.0, sigma=σ, rng=rng)

    # sample it
    sample = gaussian.sample()

    # compute the density
    x = 0
    density = gaussian.density(x)
    assert density == 1/sqrt(2*pi*σ**2) * exp(-x**2/ (2*σ**2))

    # make a vector
    v = gsl.vector(1000)
    # fill it with random numbers
    gaussian.vector(vector=v)

    # make a matrix
    m = gsl.matrix(shape=(50, 200))
    # fill it with random numbers
    gaussian.matrix(matrix=m)

    return gaussian


# main
if __name__ == "__main__":
    test()


# end of file
