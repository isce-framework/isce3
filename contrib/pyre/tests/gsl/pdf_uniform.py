#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the uniform pdf
"""


def test():
    # access the package
    import gsl

    # the support of the distribution
    support = (-1,1)
    # build a random number generator
    rng = gsl.rng()
    # build a uniform distribution
    uniform = gsl.pdf.uniform(support=support, rng=rng)

    # sample it
    sample = uniform.sample()
    assert sample >= support[0] and sample < support[1]

    density = uniform.density(0)
    assert density == 1/(support[1]-support[0])

    # make a vector
    v = gsl.vector(1000)
    # fill it with random numbers
    uniform.vector(vector=v)

    # make a matrix
    m = gsl.matrix(shape=(100, 100))
    # fill it with random numbers
    uniform.matrix(matrix=m)

    return uniform


# main
if __name__ == "__main__":
    test()


# end of file
