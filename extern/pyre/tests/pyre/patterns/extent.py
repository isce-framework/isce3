#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that extent awareness tracks the extent of classes correctly
"""


def create_instances():
    """build some instances"""

    from pyre.patterns.ExtentAware import ExtentAware

    class base(metaclass=ExtentAware):
        """base"""

    class derived(base):
        """derived"""

    class root(base, pyre_extentRoot=True):
        """keeps its own extent"""

    b1 = base()
    b2 = base()
    d1 = derived()
    d2 = derived()
    r1 = root()
    r2 = root()

    # print("b1:", b1)
    # print("b2:", b2)
    # print("d1:", d1)
    # print("d2:", d2)
    # print("r1:", r1)
    # print("r2:", r2)
    # print({ref for ref in base._pyre_extent})
    # print({ref for ref in derived._pyre_extent})
    # print({ref for ref in root._pyre_extent})
    assert set(base._pyre_extent) == { b1, b2, d1, d2 }
    assert set(root._pyre_extent) == { r1, r2 }

    return base, derived, root


def test():
    # make some instances
    base, derived, root = create_instances()

    # verify that they were destroyed when they went out of scope
    # print(set(base._pyre_extent))
    # print(set(derived._pyre_extent))
    assert set(base._pyre_extent) == set()
    assert set(root._pyre_extent) == set()
    assert base._pyre_extent is derived._pyre_extent
    assert base._pyre_extent is not root._pyre_extent

    return base, derived, root


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
