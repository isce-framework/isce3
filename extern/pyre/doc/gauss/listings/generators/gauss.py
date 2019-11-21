#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


def gauss():
    """
    The driver for the generator based implementation
    """
    from Disk import Disk
    from Mersenne import Mersenne

    # inputs
    N = 10**5
    box = [(0,1), (0,1)]
    # the point cloud
    cloud = Mersenne()
    # the region of integration
    disk = Disk(center=(0,0), radius=1)

    # the integration algorithm
    # build the point sample
    sample = cloud.points(N, box)
    # count the interior points
    interior = count(disk.interior(sample))

    # print the estimate of π
    print("π: {:.8f}".format(4*interior/N))
    return


def count(iterable): #@\label{line:driver:generators:count}@
    """
    Count the entries of iterable
    """
    counter = 0
    for item in iterable:
        counter += 1
    return counter


# main
if __name__ == "__main__":
    gauss()

# end of file
