#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that nodes can be instantiated and that their limited interface works as advertised
"""


def test():
    # get hold of the constructor
    from pyre.filesystem.Node import Node as node

    # fake a filesystem
    class filesystem: pass

    # build a node and return it
    return node(filesystem=filesystem())


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
