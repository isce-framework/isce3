#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that protocol family names are recorded correctly
"""


def test():
    import pyre

    class protocol(pyre.protocol, family="test.protocols.trivial"):
        """a trivial protocol"""

    # check the family
    assert protocol.pyre_family() == "test.protocols.trivial"
    # get the package
    package = protocol.pyre_package()
    # check the package name
    assert package.name == "test"
    # verify that the protocol is registered
    assert protocol in package.protocols

    # all done
    return protocol


# main
if __name__ == "__main__":
    test()


# end of file
