#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that we can create virtual filesystem instances
"""


def test(interactive=False): # switch to True to see the dumps
    # support
    import pyre.primitives
    # my package
    import pyre.filesystem

    # create a virtual filesystem
    fs = pyre.filesystem.virtual()

    # create a folder a few levels down from the root
    mpath = pyre.primitives.path("/home/users/mga")
    mga = fs.folder()
    fs[mpath] = mga
    # check that we can retrieve it
    assert mga == fs[mpath]

    # add a subfolder
    tpath = pyre.primitives.path('dv/tools')
    tools = fs.folder()
    fs[mpath / tpath] = tools

    # and retrieve it
    assert fs[mpath / tpath] == mga[tpath]

    # add a node
    hello = fs.node()
    tools["hello.py"] = hello

    # dump the contents
    fs.dump(interactive)

    return fs


# main
if __name__ == "__main__":
    import gc
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
