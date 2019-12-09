#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the pickler factory is accessible
"""


def test():
    # access the package
    import pyre.ipc
    # make a pickler
    return pyre.ipc.pickler()


# main
if __name__ == "__main__":
    test()


# end of file
