#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise building subvectors out of existing vectors
"""


def test():
    # package access
    import gsl

    # pick a size
    n = 20
    # make one
    v = gsl.vector(shape=n)
    # fill it
    for i in range(n): v[i] = i
    # show me
    # print('v:')
    # v.print(format='{:6.2f}', indent=' '*4)

    # pick some parameters
    start = int(n/4)
    shape = n - start
    # make a subvector
    s = v.view(start=start, shape=shape)
    # show me
    # print('s:')
    # s.print(format='{:6.2f}', indent=' '*4)

    # check the length
    assert len(s) == shape
    # check the contents
    for i in range(shape):
        assert s[i] == start + i

    # now modify
    s.fill(0)
    # and check
    for i in range(shape):
        assert v[start+i] == 0


    # all done
    return


# main
if __name__ == '__main__':
    test()

# end of file
