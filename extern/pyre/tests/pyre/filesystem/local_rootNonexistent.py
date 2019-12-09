#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that attempts to create local filesystems with nonexistent roots fails as expected
"""


def test():
    # support
    import pyre.primitives
    # my package
    import pyre.filesystem

    # attempt to
    try:
        # make a filesystem rooted at a non-existent directory
        pyre.filesystem.local(root="/@")
        # which should fail, hence we can't reach here
        assert False
    # if it fails as expected
    except pyre.filesystem.MountPointError as error:
        # check that the error message is what i expect
        assert str(error) == "error while mounting '/@': mount point not found"

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
