#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise component class configuration in the presence of inheritance
"""


def declare():
    import pyre

    # declare a component
    class base(pyre.component, family="sample.base"):
        """a base component"""
        common = pyre.properties.str(default="base")

    # derive another one from it
    class intermediate(base):
        """an intermediate component in the hierarchy that doesn't declare a family"""
        middle = pyre.properties.str(default="intermediate")

    # and a final one
    class derived(intermediate, family="sample.derived"):
        """a derived component"""
        extra = pyre.properties.str(default="derived")

    return base, intermediate, derived


def test():
    base, intermediate, derived = declare()
    # check that the settings were read properly
    # for base
    assert base.common == "base - common"
    # for intermediate
    assert intermediate.common == "base - common"
    assert intermediate.middle == "intermediate"
    # for derived
    assert derived.common == "derived - common"
    assert derived.middle == "intermediate"
    assert derived.extra == "derived - extra"
    # and return the component classes
    return base, intermediate, derived


# main
if __name__ == "__main__":
    test()


# end of file
