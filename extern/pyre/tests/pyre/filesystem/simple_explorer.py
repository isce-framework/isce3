#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the simple filesystem explorer
"""


def test():
    # my package
    import pyre.filesystem

    # make a filesystem
    fs = pyre.filesystem.virtual()
    # create a couple of nodes and insert them into the filesystem
    fs["/home/users/mga/dv/tools/bin/hello"] = fs.node()
    fs["/home/users/mga/dv/tools/lib/libhello.a"] = fs.node()

    # explore
    explorer = pyre.filesystem.simpleExplorer()
    contents = list(explorer.explore(node=fs, label='/'))
    # for line in contents: print(line)

    # check
    assert contents == [
        "(d) /",
        "  (d) home",
        "    (d) users",
        "      (d) mga",
        "        (d) dv",
        "          (d) tools",
        "            (d) bin",
        "              (f) hello",
        "            (d) lib",
        "              (f) libhello.a",
        ]

    return fs, explorer


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
