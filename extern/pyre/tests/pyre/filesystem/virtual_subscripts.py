#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify node access in folders using the subscript notation
"""


def test():
    import pyre.filesystem

    # build a folder
    root = pyre.filesystem.virtual()
    # and a node
    mga = root.node()

    # add it to the folder
    root["/home/users/mga"] = mga

    # check that it was done correctly
    assert len(root.contents) == 1
    assert "home" in root.contents

    home = root["home"]
    assert len(home.contents) == 1
    assert "users" in home.contents

    users = home["users"]
    assert len(users.contents) == 1
    assert "mga" in users.contents

    assert users["mga"] is mga

    # now look for it and make sure we got the same node
    assert root["/home/users/mga"] is mga

    # all done
    return root


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
