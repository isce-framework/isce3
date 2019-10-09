#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise loading a configuration file explicitly
"""


def test():
    # access the framework
    import pyre

    # access the nameserver
    ns = pyre.executive.nameserver
    # and the fileserver
    fs = pyre.executive.fileserver

    # load the configuration file
    pyre.loadConfiguration('vfs:{}/sample.cfg'.format(fs.STARTUP_DIR))


    # verify that the configuration settings were read properly
    assert ns["package.home"] == "home"
    assert ns["package.prefix"] == "prefix"
    assert ns["package.user.name"] == "michael a.g. aïvázis"
    assert ns["package.user.email"] == "michael.aivazis@orthologue.com"
    assert ns["package.user.affiliation"] == "orthologue"

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
