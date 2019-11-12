#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the internet address parser
"""


def test():
    # get the package parts
    from pyre.descriptors import inet

    # make one
    inet = inet()

    # positive tests
    # ip4 with full information
    address = inet.coerce("ip4:pyre.orthologue.com:50000")
    assert address.family == inet.ipv4.family
    assert address.host == "pyre.orthologue.com"
    assert address.port == 50000
    assert address.value == ("pyre.orthologue.com", 50000)

    # ip4 with no family
    address = inet.coerce("pyre.orthologue.com:50000")
    assert address.family == inet.ipv4.family
    assert address.host == "pyre.orthologue.com"
    assert address.port == 50000
    assert address.value == ("pyre.orthologue.com", 50000)

    # ip4 with no port
    address = inet.coerce("ip4:pyre.orthologue.com")
    assert address.family == inet.ipv4.family
    assert address.host == "pyre.orthologue.com"
    assert address.port == 0
    assert address.value == ("pyre.orthologue.com", 0)

    # ip4 with no family or port
    address = inet.coerce("pyre.orthologue.com")
    assert address.family == inet.ipv4.family
    assert address.host == "pyre.orthologue.com"
    assert address.port == 0
    assert address.value == ("pyre.orthologue.com", 0)

    # unix
    address = inet.coerce("unix:/tmp/.s.5394")
    assert address.family == inet.unix.family
    assert address.path == "/tmp/.s.5394"
    assert address.value == "/tmp/.s.5394"

    address = inet.coerce("local:/tmp/.s.5394")
    assert address.family == inet.unix.family
    assert address.path == "/tmp/.s.5394"
    assert address.value == "/tmp/.s.5394"

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
