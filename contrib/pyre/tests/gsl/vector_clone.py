#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Fill a vector with a constant value
"""


def test():
    # package access
    import gsl
    # make a vector
    v = gsl.vector(shape=100)
    # set it to some value
    v.random(pdf=gsl.pdf.uniform(support=(-1,1), rng=gsl.rng()))

    # clone it
    u = v.clone()
    # and check it
    assert u == v

    # all done
    return v, u


# main
if __name__ == "__main__":
    test()


# end of file
