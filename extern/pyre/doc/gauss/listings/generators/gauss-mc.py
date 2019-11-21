#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import operator, functools

def gauss():
    """
    The driver for the generator based implementation
    """
    from Disk import Disk
    from Gaussian import Gaussian
    from Mersenne import Mersenne

    # inputs
    N = 10**5
    box = [(-1,1), (-1,1)]
    B = functools.reduce(operator.mul, ((right-left) for left,right in zip(*box)))#@\label{line:mc:volume}@
    # the point cloud generator
    generator = Mersenne()
    # the region of integration
    disk = Disk(center=(0,0), radius=1)
    # the integrand
    gaussian = Gaussian(mean=(0,0), spread=1/3)

    # the integration algorithm
    # build the point sample
    sample = generator.points(N, box)
    # select the interior points
    interior = disk.interior(sample)
    # compute the integral
    integral = B/N * sum(gaussian.eval(interior))#@\label{line:mc:integral}@

    # print the estimate of the integral
    print("integral: {:.8f}".format(integral))
    return


# main
if __name__ == "__main__":
    gauss()

# end of file
