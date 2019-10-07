#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Assemble an integration entirely through a local configuration file
"""


def test():
    # load the local configuration file
    # import pyre
    # pyre.executive.configurator.dump()
    # get the montecarlo integrator from the gauss package
    import gauss
    mc = gauss.integrators.montecarlo(name="mc-π")

    # print the state of the integrator
    # print("integrator {.pyre_name!r}:".format(mc))
    # print("  samples: {.samples!r}".format(mc))
    # print("  box: {.box!r}".format(mc))
    # print("    name: {.box.pyre_name!r}".format(mc))
    # print("    intervals: {.box.intervals!r}".format(mc))
    # print("  mesh: {.mesh!r}".format(mc))
    # print("    name: {.mesh.pyre_name!r}".format(mc))
    # print("  region: {.region!r}".format(mc))
    # print("    name: {.region.pyre_name!r}".format(mc))
    # print("  integrand: {.integrand!r}".format(mc))
    # print("    name: {.integrand.pyre_name!r}".format(mc))

    # integrate to get the area of a quarter circle
    integral = mc.integrate()
    # the correct answer
    from math import pi as π
    correct = π/4

    # check the calculation
    assert abs(integral - correct)/correct < 1.0e-2
    # return the integrator
    return mc


# main
if __name__ == "__main__":
    test()


# end of file
