#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the package manager is accessible
"""


def test():
    # access to the merlin executive
    from merlin import merlin

    # get the package manager
    packages = merlin.pyre_host.packager
    # check it is accessible
    assert packages

    # and return
    return packages


# main
if __name__ == "__main__":
    test()


# end of file
