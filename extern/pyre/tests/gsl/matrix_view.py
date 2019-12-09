#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#



def test():
    # package access
    import gsl
    # pick a size
    n = 4
    # make one
    m = gsl.matrix(shape=(n,n))
    # fill it
    for i in range(n):
        for j in range(n):
            m[i,j] = i*n + j
    # show me
    # print('m:')
    # m.print(format='{:6.2f}', indent=' '*4)

    # pick some parameters
    start = (1,1)
    shape = (2,2)
    # make a submatrix
    v = m.view(start=start, shape=shape)
    # show me
    # print('v:')
    # v.print(format='{:6.2f}', indent=' '*4)

    # verify the shape
    assert v.shape == shape
    # verify the contents
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert v[i,j] == m[i+start[0], j+start[1]]

    # now modify
    v.fill(0)
    # show me
    # print('m:')
    # m.print(format='{:6.2f}', indent=' '*4)
    # and check
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert m[i+start[0],j+start[1]] == 0

    # all done
    return


# main
if __name__ == '__main__':
    test()

# end of file
