#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify various configuration methods
"""


# externals
import pyre

# the protocols
class functor(pyre.protocol, family="quad.functors"):
    @pyre.provides
    def eval(self, z):
        """evaluate a function at the given argument {z}"""

# components
# functors
class const(pyre.component, family="quad.functors.const"):
    c = pyre.properties.float()

    @pyre.export
    def eval(self, z): return self.c

class line(pyre.component, family="quad.functors.line"):
    α = pyre.properties.float()
    β = pyre.properties.float()

    @pyre.export
    def eval(self, z): return self.α * z + self.β


class integrator(pyre.component, family="quad.integrator"):
    integrand = functor(default=const)


# the tests
def test():

    # print the configuration
    # pyre.executive.configurator.dump(pattern='quad')
    # for error in pyre.executive.errors: print(error)

    # check the class defaults from the configuration file
    # const
    assert const.c == 1.0
    # line
    assert line.α == 1.0
    assert line.β == 2.0
    # integrator
    assert integrator.integrand.pyre_family() == line.pyre_family()

    # instantiations
    zero = const(name='zero')
    assert zero.c == 0

    two = const(name='two')
    assert two.c == 2.0

    # a default integrator
    nameless = integrator(name='nameless')
    assert nameless.pyre_name == 'nameless'
    assert nameless.pyre_family == integrator.pyre_family
    assert nameless.integrand.pyre_name == 'nameless.integrand'
    assert nameless.integrand.pyre_family() == line.pyre_family()
    assert nameless.integrand.α == line.α
    assert nameless.integrand.β == line.β

    # a named one
    special = integrator(name='special')
    assert special.pyre_name == 'special'
    assert special.pyre_family == integrator.pyre_family
    assert special.integrand.pyre_name == 'special.integrand'
    assert special.integrand.pyre_family() == const.pyre_family()
    assert special.integrand.c == 3.0

    # another named one
    qualified = integrator(name='qualified')
    assert qualified.pyre_name == 'qualified'
    assert qualified.pyre_family == integrator.pyre_family
    assert qualified.integrand.pyre_name == 'qualified.integrand'
    assert qualified.integrand.pyre_family() == line.pyre_family()
    assert qualified.integrand.α == 0.5
    assert qualified.integrand.β == 1.5

    # a named one with an explicitly named integrand
    explicit = integrator(name='explicit')
    assert explicit.pyre_name == 'explicit'
    assert explicit.pyre_family == integrator.pyre_family
    assert explicit.integrand.pyre_name == 'two'
    assert explicit.integrand.pyre_family == const.pyre_family
    assert explicit.integrand.c == 2.0

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
