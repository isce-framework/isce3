#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that compatibility among components is detected correctly
"""


def test():
    import pyre

    # declare a couple of components
    class base(pyre.component):
        """the base component"""
        common = pyre.property()

    class derived(base):
        """a derived one, so automatically compatible"""
        extra = pyre.property()

    class ok(pyre.component):
        """one that doesn't derive but has the right public component"""
        common = pyre.property()

    class notok(pyre.component):
        """one that doesn't provide the right public component"""
        what = pyre.property()

    class badtype(pyre.component):
        """one that has the right trait but of the wrong type"""
        @pyre.provides
        def common(self):
            """method, not property"""

    class shadow(base):
        """one that derives but shadows the trait in an incompatible way"""
        @pyre.provides
        def common(self):
            """method, not property"""

    # compatibility checks
    # the ones that should succeed
    assert derived.pyre_isCompatible(base)
    assert ok.pyre_isCompatible(base)
    assert derived.pyre_isCompatible(ok)
    # and the ones that should fail
    assert not ok.pyre_isCompatible(derived)
    assert not notok.pyre_isCompatible(base)
    assert not notok.pyre_isCompatible(derived)
    assert not notok.pyre_isCompatible(ok)
    assert not badtype.pyre_isCompatible(base)
    assert not badtype.pyre_isCompatible(derived)
    assert not badtype.pyre_isCompatible(ok)
    assert not shadow.pyre_isCompatible(base)
    assert not shadow.pyre_isCompatible(derived)
    assert not shadow.pyre_isCompatible(ok)

    return base, derived, ok, notok, badtype, shadow


# main
if __name__ == "__main__":
    test()


# end of file
