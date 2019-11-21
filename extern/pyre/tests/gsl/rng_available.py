#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that we can identify available generators
"""


def test():
    import gsl
    # pick an algorithm to use
    algorithm = 'ranlxs2'
    # verify it is in the set of known names
    assert algorithm in gsl.rng.available
    # and return it
    return gsl.rng


# main
if __name__ == "__main__":
    test()


# end of file
