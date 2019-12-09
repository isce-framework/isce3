#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that folders can be instantiated and that their limited interface works as advertised
"""


def test():
    from pyre.filesystem.Folder import Folder as folder

    # fake a filesystem
    class filesystem: pass
    # build a fake filesystem
    fs = filesystem()

    # build a folder
    root = folder(filesystem=fs)
    # and some nodes
    usr = root.node()
    tmp = root.node()
    home = root.node()
    # add them to the folder
    root.contents["usr"] = usr
    root.contents["tmp"] = tmp
    root.contents["home"] = home

    # count the children
    assert len(root.contents) == 3

    # access the individual nodes
    assert usr == root.contents["usr"]
    assert tmp == root.contents["tmp"]
    assert home == root.contents["home"]

    # all done
    return root


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
