#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Create and dump a local filesystem
"""


def test():
    import pyre.filesystem

    home = pyre.filesystem.local(root="../..")
    home.discover()
    # print('\n'.join(home.dump()))

    return home


# main
if __name__ == "__main__":
    # request debugging support for the pyre.calc package
    pyre_debug = { "pyre.filesystem" }
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()

    # check that the nodes were all destroyed
    from pyre.filesystem.Node import Node
    # print("Node extent:", len(Node._pyre_extent))
    assert len(Node._pyre_extent) == 0


# end of file
