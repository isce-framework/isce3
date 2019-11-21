#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate the known generators
"""


def test():
    # access the package
    import gsl

    # pick an algorithm
    algorithm = 'ranlxs2'

    # instantiate the rng
    rng = gsl.rng(algorithm=algorithm)
    # check its name
    assert rng.algorithm == algorithm
    # get its range
    low, high = rng.range
    # grab a random number
    sample = rng.int()
    # check that it falls within the range
    assert sample >= low and sample < high

    # all done
    return gsl.rng


# main
if __name__ == "__main__":
    test()


# end of file
