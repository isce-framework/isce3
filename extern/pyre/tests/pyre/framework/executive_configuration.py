#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that a sample configuration file can be ingested correctly
"""


def test():
    import pyre
    # build the executive
    executive = pyre.executive
    # verify the right parts were built
    assert executive.linker is not None
    assert executive.nameserver is not None
    assert executive.fileserver is not None
    assert executive.configurator is not None
    # load a configuration file
    executive.loadConfiguration(uri="sample.pml", priority=executive.priority.user)
    # get the nameserver
    ns = executive.nameserver
    # check that all is as expected
    assert ns["package.home"] == "home"
    assert ns["package.prefix"] == "prefix"
    assert ns["package.user.name"] == "michael a.g. aïvázis"
    assert ns["package.user.email"] == "michael.aivazis@orthologue.com"
    assert ns["package.user.affiliation"] == "orthologue"

    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
