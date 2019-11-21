#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that declarations of trivial protocols produce the expected layout
"""


def test():
    import pyre

    # declare
    class protocol(pyre.protocol):
        """a trivial protocol"""

    # check the basics
    assert protocol.__name__ == "protocol"
    assert protocol.__bases__ == (pyre.protocol,)

    # did I get a key
    assert protocol.pyre_key is None # since i didn't specify my family name
    # check the layout
    assert protocol.pyre_namemap == {}
    assert protocol.pyre_localTraits == ()
    assert protocol.pyre_inheritedTraits == ()
    assert protocol.pyre_pedigree == (protocol, pyre.protocol)
    assert protocol.pyre_internal == False

    # exercise the configurable interface
    assert list(protocol.pyre_traits()) == []
    assert protocol.pyre_isCompatible(protocol)

    return protocol


# main
if __name__ == "__main__":
    test()


# end of file
