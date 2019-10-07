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

    # access to the package
    import pyre
    # instantiate a script and return it
    return pyre.shells.script()(name="test")


# main
if __name__ == "__main__":
    test()


# end of file
