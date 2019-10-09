#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify node insertion in folders

In this test, we first create the parent of the target node separately, followed by an
insertion of the target node through its absolute path. We then check that the tree structure
is as expected
"""


def test():
    # support
    import pyre.primitives
    # my package
    import pyre.filesystem

    # build a folder
    root = pyre.filesystem.virtual()
    # and a couple of nodes
    mga = root.node()
    users = root.folder()

    # add them to the folder
    root._insert(node=users, uri=pyre.primitives.path("/home/users"))
    root._insert(node=mga, uri=pyre.primitives.path("/home/users/mga"))

    # check that it was done correctly
    assert len(root.contents) == 1
    assert "home" in root.contents

    home = root.contents["home"]
    assert len(home.contents) == 1
    assert "users" in home.contents

    users = home.contents["users"]
    assert len(users.contents) == 1
    assert "mga" in users.contents

    assert users.contents["mga"] is mga

    # all done
    return root


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
