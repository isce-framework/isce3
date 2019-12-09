#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the trait defaults get registered correctly
"""


def test():
    import pyre

    # declare a couple of components
    class base(pyre.component):
        """the base component"""
        common = pyre.property()
        common.default = "base.common"

    class derived(base):
        """the derived one"""
        extra = pyre.property()
        extra.default = "derived.extra"

    # check that the defaults are readable
    assert base.common == "base.common"
    assert derived.extra == "derived.extra"
    assert derived.common == "base.common"
    # check that the defaults are writeable
    base.common = "base:common"
    derived.common = "derived:common"
    derived.extra = "derived:extra"
    # check that they were recorded properly
    assert base.common == "base:common"
    assert derived.common == "derived:common"
    assert derived.extra == "derived:extra"

    return base, derived


# main
if __name__ == "__main__":
    test()


# end of file
