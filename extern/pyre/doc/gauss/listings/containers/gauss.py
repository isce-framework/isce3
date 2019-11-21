#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


def gauss():
    """
    The driver for the container based implementation
    """
    from Disk import Disk
    from Mersenne import Mersenne

    # inputs
    N = 10**5
    box = [(0,1), (0,1)]
    # the point cloud generator
    cloud = Mersenne()
    # the region of integration
    disk = Disk(center=(0,0), radius=1)

    # the integration algorithm
    # build the point sample
    sample = cloud.points(N, box)
    # count the interior points
    interior = len(disk.interior(sample))

    # print the estimate of π
    print("π: {:.8f}".format(4*interior/N))
    return


# main
if __name__ == "__main__":
    gauss()

# end of file
