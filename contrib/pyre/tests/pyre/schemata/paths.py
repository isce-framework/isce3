#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that path conversions work as expected
"""


def test():
    # the trait descriptors
    import pyre.schemata

    # create a descriptor
    path = pyre.schemata.path()

    # the default is the current working directory
    cwd = path.coerce('')
    # which always exists
    assert cwd.exists()
    # and has a standard rep
    assert str(cwd) == '.'

    # resolve it
    cwd = cwd.resolve()
    # verify that it is the same path as the actual {cwd}
    assert cwd == path.coerce(value=path.cwd())

    # get its parent
    parent = cwd.parent
    # and verify that it is a child of its parent
    assert cwd.name in (folder.name for folder in parent.contents)

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
