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
    # create a few nodes and insert them into the filesystem
    fs["/home/users/mga/tools/bin/hello"] = fs.node()
    fs["/home/users/mga/tools/bin/goodbye"] = fs.node()
    fs["/home/users/mga/tools/lib/libhello.a"] = fs.node()
    fs["/home/users/mga/tools/lib/libgoodbye.a"] = fs.node()
    fs["/home/users/mga/dv/pyre-1.0/packages/pyre/__init__.py"] = fs.node()

    # explore
    explorer = pyre.filesystem.treeExplorer()
    contents = list(explorer.explore(node=fs, label='/'))
    # for line in contents: print(line)

    # check
    assert contents == [
        "/ (d)",
        "`- home (d)",
        "   `- users (d)",
        "      `- mga (d)",
        "         +- dv (d)",
        "         |  `- pyre-1.0 (d)",
        "         |     `- packages (d)",
        "         |        `- pyre (d)",
        "         |           `- __init__.py (f)",
        "         `- tools (d)",
        "            +- bin (d)",
        "            |  +- goodbye (f)",
        "            |  `- hello (f)",
        "            `- lib (d)",
        "               +- libgoodbye.a (f)",
        "               `- libhello.a (f)",
        ]

    return fs, explorer


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
