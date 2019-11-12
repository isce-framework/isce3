#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify the linker is accessible through the executive
"""


def test():
    import pyre
    # build the executive
    executive = pyre.executive

    # access the linker
    assert executive.linker is not None

    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
