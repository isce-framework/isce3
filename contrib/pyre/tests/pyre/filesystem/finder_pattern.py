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
    fs["/home/users/mga/dv/tools/src/hello.c"] = fs.node()
    fs["/home/users/mga/dv/tools/src/hello.h"] = fs.node()
    fs["/home/users/mga/dv/tools/lib/libhello.a"] = fs.node()

    # dump the contents
    # print('\n'.join(fs.dump()))

    # explore
    finder = pyre.filesystem.finder()
    # collect the contents of interest
    contents = list(sorted(
        str(node.uri)
        for node, *_ in finder.explore(folder=fs, pattern=r".*\.h")))

    # check
    assert contents == [
        "/home/users/mga/dv/tools/src/hello.h",
        ]

    return fs, finder


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
