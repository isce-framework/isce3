#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercises the fileserver
"""


def test():
    import pyre
    # access the fileserver
    fs = pyre.executive.fileserver
    # make sure we got one
    assert fs is not None

    # initialize it
    fs.initializeNamespace()

    # get hold of the package node
    packages = fs[fs.PACKAGES_DIR]
    assert packages is not None

    # get hold of the user node
    user = fs[fs.USER_DIR]
    assert user is not None

    # dump the filesystem
    # print('\n'.join(fs.dump()))

    # all done
    return fs


# main
if __name__ == "__main__":
    test()


# end of file
