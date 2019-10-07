#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Make a matrix and fill it with random values
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(100,50))
    # set it to random values
    m.random(pdf=gsl.pdf.gaussian(mean=0.0, sigma=2, rng=gsl.rng()))
    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
