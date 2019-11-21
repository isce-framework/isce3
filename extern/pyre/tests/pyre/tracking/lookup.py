#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the lookup locator returns the correct location tag
"""


def lookup():
    import pyre

    key = pyre.executive.nameserver.hash('pyre')
    locator = pyre.tracking.lookup(description="while looking up", key=key)

    assert str(locator) == "while looking up package 'pyre'"

    return locator


# main
if __name__ == "__main__":
    # do...
    lookup()


# end of file
