#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that bad component descriptors raise the correct exceptions
"""


def test():
    import pyre
    executive = pyre.executive

    # attempt to retrieve a non-existent component descriptor from the python path
    try:
        unkown, = executive.resolve(uri="import:not-there/unknown")
        assert False
    except ValueError:
        pass
    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
