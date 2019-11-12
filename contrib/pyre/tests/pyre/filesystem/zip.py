#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify creation of filesystems based on zipfiles
"""


def test(interactive=False): # set to True to see the dump
    import os
    import zipfile
    import pyre.filesystem


    # the name of the zipfile
    archive = "sample.zip"
    # open it as a filesystem
    home = pyre.filesystem.zip(root=archive)
    # ingest the contents
    home.discover()
    home.dump(interactive)

    # all done
    return home


# main
if __name__ == "__main__":
    # request debugging support for the pyre.calc package
    pyre_debug = { "pyre.filesystem" }
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()

    # check that the nodes were all destroyed
    from pyre.filesystem.Node import Node
    # print("Node extent:", len(Node._pyre_extent))
    assert len(Node._pyre_extent) == 0


# end of file
