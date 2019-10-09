#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: make sure we can instantiate variable slots
"""

def test():
    # for the locator
    import pyre.tracking
    # get the slot class
    from pyre.framework.Slot import Slot

    # make a key
    key = None
    # and a value
    value = 4

    # make a slot
    return Slot.variable(key=key, value=value)


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
