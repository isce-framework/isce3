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
    # my package
    import pyre.filesystem

    # attempt to
    try:
        # mount a zip filesystem at nonexistent location
        pyre.filesystem.zip(root="/@")
        # which should fail so we can't get here
        assert False
    # if it fails as expected
    except pyre.filesystem.MountPointError as error:
        # check that the error message is corerct
        assert str(error) == "error while mounting '/@': mount point not found"

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
