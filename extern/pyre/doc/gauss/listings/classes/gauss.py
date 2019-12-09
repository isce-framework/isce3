#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

def gauss():
    """
    The driver for the object oriented solution
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
    interior = 0
    for i in range(N):
        point = cloud.point(box)
        if disk.interior(point):
            interior += 1
    # print the estimate of π
    print("π: {:.8f}".format(4*interior/N))
    return

# main
if __name__ == "__main__":
    gauss()

# end of file
