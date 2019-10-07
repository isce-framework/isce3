#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate a script
"""


def test():
    # access the package
    import pyre
    # build a fork and return it
    return pyre.shells.fork()(name="test")


# main
if __name__ == "__main__":
    test()


# end of file
