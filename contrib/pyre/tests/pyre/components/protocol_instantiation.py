#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that an exception gets raised when a protocol is instantiated
"""


def test():
    import pyre

    # declare
    class protocol(pyre.protocol):
        """a trivial protocol"""
        p = pyre.property()

    # instantiate
    facility = protocol()
    # verify the result is a facility
    assert isinstance(facility, pyre.facility)

    return protocol


# main
if __name__ == "__main__":
    test()


# end of file
