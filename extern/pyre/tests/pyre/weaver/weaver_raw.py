#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: instantiate a weaver and verify its configuration
"""


def test():
    # get the package
    import pyre.weaver
    # instantiate a weaver
    weaver = pyre.weaver.weaver(name="sanity")
    # by default, there is no language setting
    assert weaver.language == None
    # and return it
    return weaver


# main
if __name__ == "__main__":
    test()


# end of file
