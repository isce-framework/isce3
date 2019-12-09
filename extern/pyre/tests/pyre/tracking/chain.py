#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that locators can be chained correctly
"""


def script():
    import pyre.tracking

    first = pyre.tracking.simple(source="first")
    second = pyre.tracking.simple(source="second")
    chain = pyre.tracking.chain(this=first, next=second)

    assert str(chain) == "first, second"

    return chain


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    script()


# end of file
