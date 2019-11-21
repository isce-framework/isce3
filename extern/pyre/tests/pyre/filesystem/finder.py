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
    import pyre.filesystem

    # make a filesystem
    fs = pyre.filesystem.virtual()
    # create a couple of nodes and insert them into the filesystem
    fs["/home/users/mga/dv/tools/bin/hello"] = fs.node()
    fs["/home/users/mga/dv/tools/lib/libhello.a"] = fs.node()

    # explore
    finder = pyre.filesystem.finder()
    # get the contents
    contents = list(sorted(
        str(node.uri)
        for node, *_ in finder.explore(fs)))
    # show me
    # for line in contents: print(line)

    # check
    assert contents == [
        "/",
        "/home",
        "/home/users",
        "/home/users/mga",
        "/home/users/mga/dv",
        "/home/users/mga/dv/tools",
        "/home/users/mga/dv/tools/bin",
        "/home/users/mga/dv/tools/bin/hello",
        "/home/users/mga/dv/tools/lib",
        "/home/users/mga/dv/tools/lib/libhello.a",
        ]

    return fs, finder


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
