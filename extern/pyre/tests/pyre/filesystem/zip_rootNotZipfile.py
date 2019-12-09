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
    import pyre.filesystem

    dummy = "./local_rootNotDirectory.py"
    try:
        pyre.filesystem.zip(root=dummy)
        assert False
    except pyre.filesystem.MountPointError as error:
        import os
        target = os.path.abspath(dummy)
        assert str(error) == (
            "error while mounting {0!r}: mount point is not a zipfile".format(target))

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
