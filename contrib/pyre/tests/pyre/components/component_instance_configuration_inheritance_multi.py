#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Validate component configuration in the presence of multiple inheritance
"""


def declare():
    import pyre

    # declare a component
    class base(pyre.component, family="sample.base.multi"):
        """a base component"""
        common = pyre.properties.str(default="base")

    # derive another one from it
    class a1(base, family="sample.a1.multi"):
        """an intermediate component in the hierarchy that doesn't declare a family"""
        middle = pyre.properties.str(default="a1")

    class a2(base, family="sample.a2.multi"):
        """an intermediate component in the hierarchy that doesn't declare a family"""
        common = pyre.properties.str(default="a2")

    # and a final one
    class derived(a1, a2, family="sample.derived.multi"):
        """a derived component"""
        extra = pyre.properties.str(default="derived")

    return base, a1, a2, derived


def test():
    # get the declarations
    base, a1, a2, derived = declare()
    # instantiate
    d = derived(name="d")
    # check that the settings were read properly
    assert d.common == "a2 - common"
    assert d.middle == "a1 - middle"
    assert d.extra == "d - extra"
    # and return the component classes
    return d


# main
if __name__ == "__main__":
    test()


# end of file
