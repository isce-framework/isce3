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
    # check that the settings were read properly
    # for base
    assert base.common == "base - common"
    # for a1
    assert a1.common == "a1 - common"
    assert a1.middle == "a1 - middle"
    # for a2
    assert a2.common == "a2 - common"
    # for derived
    assert derived.common == "a2 - common"
    assert derived.middle == "a1 - middle"
    assert derived.extra == "derived - extra"
    # and return the component classes
    return base, a1, a2, derived


# main
if __name__ == "__main__":
    test()


# end of file
