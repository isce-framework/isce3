#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Lijun Zhu
# Caltech
# (c) 1998-2019 all rights reserved
#


"""
Exercise the uniform pdf
"""


def test():
    # access the package
    import gsl

    # build a random number generator
    rng = gsl.rng()
    # build a uniform distribution
    uniform = gsl.pdf.uniform_pos(rng=rng)

    # sample it
    sample = uniform.sample()
    # assert the sample is within (0, 1)
    assert sample > 0 and sample < 1

    density = uniform.density(0)
    assert density == 1

    # make a vector
    v = gsl.vector(1000)
    # fill it with random numbers
    uniform.vector(vector=v)
    # assert all the samples are positive
    for sample in v: assert sample > 0 and sample < 1

    # make a matrix
    m = gsl.matrix(shape=(100, 100))
    # fill it with random numbers
    uniform.matrix(matrix=m)
    # assert all the samples are positive
    for sample in m: assert sample > 0 and sample < 1

    return uniform


# main
if __name__ == "__main__":
    test()


# end of file
