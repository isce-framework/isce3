#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the spawning of daemons
"""


def test():
    # access the framework
    import pyre
    # instantiate a daemon and return it
    return pyre.shells.daemon()(name="daemon")


# main
if __name__ == "__main__":
    test()


# end of file
