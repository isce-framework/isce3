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
    import gsl

    # print(" ++ rng_allocate:")
    # loop over all the known algorithms
    for algorithm in gsl.rng.available:
        # print("    {}".format(algorithm))
        # instantiate the rng
        rng = gsl.rng(algorithm=algorithm)
        # check
        assert rng.algorithm == algorithm

    # all done
    return gsl.rng


# main
if __name__ == "__main__":
    test()


# end of file
