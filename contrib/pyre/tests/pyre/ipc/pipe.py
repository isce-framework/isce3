#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the pipe factory is accessible
"""


def test():
    # get the package
    import pyre.ipc
    # make a pair of pipes
    return pyre.ipc.pipe()


# main
if __name__ == "__main__":
    test()


# end of file
