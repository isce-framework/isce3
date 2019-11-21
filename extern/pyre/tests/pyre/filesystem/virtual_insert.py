#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify node insertion in folders
"""


def test():
    # support
    import pyre.primitives
    # my package
    import pyre.filesystem

    # build a virtual filesystem
    fs = pyre.filesystem.virtual()

    # build a node
    mga = fs.node()

    # add it to the folder
    fs._insert(uri=pyre.primitives.path("/home/users/mga"), node=mga)

    # check that it was done correctly
    assert len(fs.contents) == 1
    assert "home" in fs.contents

    home = fs.contents["home"]
    assert len(home.contents) == 1
    assert "users" in home.contents

    users = home.contents["users"]
    assert len(users.contents) == 1
    assert "mga" in users.contents

    assert users.contents["mga"] is mga

    # all done
    return fs


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
