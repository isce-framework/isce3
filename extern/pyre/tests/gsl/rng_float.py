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
    # grab a random number
    sample = rng.float()
    # check that it falls within the range
    assert sample >= 0 and sample < 1

    # all done
    return gsl.rng


# main
if __name__ == "__main__":
    test()


# end of file
