#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Lijun Zhu
# Caltech
# (c) 1998-2019 all rights reserved
#


"""
Test stats_correlation
"""


def test():
    # package access
    import gsl
    # make two vectors
    length = 100
    v1 = gsl.vector(shape=length)
    v2 = gsl.vector(shape=length)
    # set them to random values
    rng = gsl.rng()
    v1.random(pdf=gsl.pdf.uniform(support=(-1,1), rng=rng))
    v2.random(pdf=gsl.pdf.uniform(support=(-1,1), rng=rng))

    # call correlation
    correlation = gsl.stats.correlation(v1, v2)

    # correlation of a vector with itself should be one
    assert gsl.stats.correlation(v1, v1) == 1.0

    # all done
    return correlation


# main
if __name__ == "__main__":
    test()


# end of file
