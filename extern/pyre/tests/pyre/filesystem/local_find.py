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

    # build a filesystem
    tests = pyre.filesystem.local(root="..").discover()

    # look for this file
    this = tests["filesystem/local_find.py"]
    # make sure we got it
    assert this is not None
    # and that it is the right uri
    assert this.uri == pyre.primitives.path(__file__).resolve()

    # all done
    return this


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
