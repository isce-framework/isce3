#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the metadata associated with node are maintained properly
"""


def test():
    # support
    import pyre.primitives
    # my package
    import pyre.filesystem

    # build a virtual filesystem
    root = pyre.filesystem.virtual()
    # and a couple of nodes
    root['home/users'] = root.folder()
    root['home/users/mga'] = root.folder()

    # check their uris
    assert str(root['home/users'].uri) == '/home/users'
    assert str(root['home/users/mga'].uri) == '/home/users/mga'

    # all done
    return root


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
