#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise trait aliases
"""


import pyre


def declare():

    class gaussian(pyre.component, family="functor"):
        """a representation of a gaussian function"""

        # properties
        mean = pyre.properties.float(default="0.01")
        mean.aliases.add("μ")

        spread = pyre.properties.float(default="0.01")
        spread.aliases.add("σ")

        # behaviors
        @pyre.export
        def eval(self, x):
            return x

    return gaussian


def test():

    ns = pyre.executive.nameserver
    # print the store before the declaration
    # print(" -- at startup:")
    # ns.dump(pattern="(functor|gaussian)")
    # get the commandline slots
    # mean,_ = ns.getNode(key=ns.hash('functor.μ'))
    # mean.dump(name='functor.μ')
    # print(" -- done")

    functor = declare()
    # check that the aliases were properly registered
    assert functor.pyre_trait("mean") == functor.pyre_trait("μ")
    assert functor.pyre_trait("spread") == functor.pyre_trait("σ")
    # print out the configuration state
    # print(" -- after the declaration:")
    # print("functor: defaults: mean={0.mean!r}, spread={0.spread!r}".format(functor))
    # ns.dump(pattern="(functor|gaussian)")
    # print(" -- done")

    # check the class defaults
    # the values come from the defaults, functor.pml in this directory, and the command line
    assert functor.mean == 0.1
    assert functor.spread == 0.54
    # reset them to something meaningful
    functor.μ = 0.0
    functor.σ = 1.0
    # verify the change
    assert functor.mean == 0.0
    assert functor.spread == 1.0

    # instantiate one
    g = functor(name="gaussian")
    # make sure the defaults were transferred correctly
    assert g.mean == 0.56
    assert g.spread == 0.10
    # use the canonical names to reconfigure
    g.mean = 1.0
    g.spread = 2.0
    # check that access through the canonical name and the alias yield the same values
    assert g.μ == g.mean
    assert g.σ == g.spread
    # use the aliases to set
    g.μ = 0.0
    g.σ = 1.0
    # and check again
    assert g.μ == g.mean
    assert g.σ == g.spread

    # check the properties
    # print("g: mean={0.mean!r}, spread={0.spread!r}".format(g))
    # ns.dump(pattern="(functor|gaussian)")

    return functor


# main
if __name__ == "__main__":
    test()


# end of file
