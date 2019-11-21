#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify searching through folders for named nodes
"""


def test():
    # support
    import pyre.primitives
    # my package
    import pyre.filesystem

    # build a folder
    root = pyre.filesystem.virtual()
    # and a node
    mga = root.node()

    # add it to the folder
    root._insert(uri=pyre.primitives.path("home/users/mga"), node=mga)

    # now retrieve it
    target = root._retrieve(uri=pyre.primitives.path("/home/users/mga"))

    # make sure it is the same node
    assert mga is target

    # all done
    return root


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
