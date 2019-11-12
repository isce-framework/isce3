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

    # build a random number generator
    rng = gsl.rng()

    # the order of the distribution
    K = 10
    # the weight vectors
    α = gsl.vector(shape=K).fill(K**(-1/2))
    # build a gaussian distribution
    dirichlet = gsl.pdf.dirichlet(alpha=α, rng=rng)

    # make a vector
    v = gsl.vector(shape=K)
    # fill it with random numbers
    v.random(pdf=dirichlet)

    # make a matrix
    m = gsl.matrix(shape=(50, K))
    # fill it with random numbers
    m.random(pdf=dirichlet)

    return dirichlet


# main
if __name__ == "__main__":
    test()


# end of file
