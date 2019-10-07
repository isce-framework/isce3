#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Create and dump a virtual filesystem
"""


def test(interactive=False): # change to True to see the dump
    # access the package
    import pyre.filesystem
    # create a virtual filesystem
    fs = pyre.filesystem.virtual()
    # create a few nodes and insert them into the filesystem
    fs["/home/users/mga/tools/bin/hello"] = fs.node()
    fs["/home/users/mga/tools/bin/goodbye"] = fs.node()
    fs["/home/users/mga/tools/lib/libhello.a"] = fs.node()
    fs["/home/users/mga/tools/lib/libgoodbye.a"] = fs.node()
    fs["/home/users/mga/dv/pyre-1.0/packages/pyre/__init__.py"] = fs.node()
    fs["/home/users/mga/dv/pyre-1.0/packages/journal/__init__.py"] = fs.node()
    # dump
    fs.dump(interactive) # change to True to see the dump

    # now clear its contents explicitly
    fs.contents = {}
    # verify that all the nodes except the filesystem itself were destroyed
    assert len(fs._pyre_extent) == 1

    # all done
    return fs


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
