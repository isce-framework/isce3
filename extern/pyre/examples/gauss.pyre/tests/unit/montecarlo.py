#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
sanity check: verify that the MonteCarlo integrator works as expected
"""


def test():
    import gauss
    from math import pi as π

    # instantiate the integrand
    one = gauss.functors.one(name="one")
    # set up the bounding box for the random number generator
    box = gauss.shapes.box(name="box")
    box.intervals = ((0,1), (0,1))
    # instantiate the region of integration
    disk = gauss.shapes.ball(name="disk")
    disk.center = (0,0)
    disk.radius = 1.0
    # instantiate the point cloud generator
    mesh = gauss.meshes.mersenne(name="mesh")

    # instantiate the integrator
    mc = gauss.integrators.montecarlo(name="mc")
    mc.samples = 10**5
    mc.box = box
    mc.mesh = mesh
    mc.region = disk
    mc.integrand = one

    # integrate
    integral = mc.integrate()

    # check the answer
    # print("π =", 4*integral)
    error = (π - 4*integral)/π
    # print("error =", π - 4*integral)
    assert error < 1.0e-2

    return mc


# main
if __name__ == "__main__":
    test()


# end of file
