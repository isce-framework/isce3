#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that protocols with properties have the expected layout
"""


def test():
    import pyre

    # declare
    class protocol(pyre.protocol):
        """a trivial protocol"""
        p = pyre.property()

    # check the basics
    assert protocol.__name__ == "protocol"
    assert protocol.__bases__ == (pyre.protocol,)
    # check the layout
    assert protocol.pyre_key is None
    assert protocol.pyre_namemap == {'p': 'p'}
    assert protocol.pyre_pedigree == (protocol, pyre.protocol)
    # traits
    localNames = ['p']
    localTraits = tuple(map(protocol.pyre_trait, localNames))
    assert protocol.pyre_localTraits == localTraits
    assert protocol.pyre_inheritedTraits == ()
    allNames = localNames + []
    allTraits = list(map(protocol.pyre_trait, allNames))
    assert list(protocol.pyre_traits()) == allTraits

    return protocol


# main
if __name__ == "__main__":
    test()


# end of file
