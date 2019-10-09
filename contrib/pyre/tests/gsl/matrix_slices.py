#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Access matrices by rows and columns
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(100,50))
    # fill it with random values
    m.random(pdf=gsl.pdf.gaussian(mean=0.0, sigma=2, rng=gsl.rng()))

    # rows
    i = 3
    # set the ith row to all {i}
    for j in range(m.columns) : m[i,j] = i
    # get the row using the interface
    row = m.getRow(i)
    # check that it is a vector
    assert row.shape == m.columns
    # full of {i}
    assert row == gsl.vector(shape=m.columns).fill(i)

    # columns
    j = 2
    # set the jth column to all {j}
    for i in range(m.rows): m[i,j] = j
    # get the column using the interface
    column = m.getColumn(j)
    # check that it is a vector
    assert column.shape == m.rows
    # full of {j}
    assert column == gsl.vector(shape=m.rows).fill(j)

    # shape
    rows = 100
    columns = 200
    # make another matrix
    m = gsl.matrix(shape=(rows,columns)).zero()

    # make a vector of ones
    ones = gsl.vector(shape=columns).fill(1.0)
    # set the middle column
    m.setRow(rows/2, ones)
    # verify it was done properly
    assert m.getRow(rows/2) == ones

    # make a vector of twos
    twos = gsl.vector(shape=rows).fill(2.0)
    # set the middle column
    m.setColumn(columns/2, twos)
    # verify it was done properly
    assert m.getColumn(columns/2) == twos

    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
