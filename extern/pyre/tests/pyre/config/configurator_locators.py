#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the configurator factory is accessible
"""


def test():
    import pyre

    # gain access to the configurator
    cfg = pyre.executive.configurator
    # load a configuration file
    pyre.loadConfiguration("sample.pml")
    cfg["sample.user.name"] = "Joe Applegate"
    # get a configuration node
    node = cfg.resolve(name="sample.user.name")
    node.dump()

    return


# main
if __name__ == "__main__":
    test()


# end of file
