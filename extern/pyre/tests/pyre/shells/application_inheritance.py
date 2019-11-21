#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the application component is accessible
"""


def test():
    # get access to the framework
    import pyre

    # declare a component
    class component(pyre.component, family='sample.component'):
        """A simple component"""
        trait = pyre.properties.str(default="trait")


    # declare a trivial application that inherits from this component
    class application(pyre.application, component, family="sample.application"):
        """A trivial pyre application"""

    # show me
    # print("{}:".format(application.pyre_family()))
    # check its traits
    # print("  local traits:")
    # print("    {}".format(tuple(trait.name for trait in application.pyre_localTraits)))
    # print("  inherited traits:")
    # print("    {}".format(tuple(trait.name for trait in application.pyre_inheritedTraits)))
    # check its inventory
    # inv = application.pyre_inventory
    # print("  inventory:")
    # for trait, slot in inv.items():
        # print("    {}: {!r}".format(trait.name, inv[trait].value))
    # print("  done")

    # check that {trait} is an {application} trait
    assert application.pyre_trait('trait')

    # now make an instance
    app = application('simple:app')
    # check that {trait} is an one of its traits
    assert app.pyre_trait('trait')

    # return the application class
    return application


# main
if __name__ == "__main__":
    test()


# end of file
