#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Lijun Zhu
# Caltech
# (c) 1998-2019 all rights reserved
#


"""
Test stats_covariance
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
    covariance = gsl.stats.covariance(v1, v2)

    # covariance between a vector and itself is its variance
    assert gsl.stats.covariance(v1, v1) == v1.variance(mean=v1.mean())

    # all done
    return covariance


# main
if __name__ == "__main__":
    test()


# end of file
