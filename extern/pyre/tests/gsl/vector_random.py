#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Fill a vector with random values
"""


def test():
    # package access
    import gsl
    # make a vector
    v = gsl.vector(shape=100)
    # set it to random values
    v.random(pdf=gsl.pdf.uniform(support=(-1,1), rng=gsl.rng()))
    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
