#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the linker
"""


def test():
    import pyre
    # get the linker
    linker = pyre.executive.linker

    # check the registered codecs
    assert tuple(linker.schemes.keys()) == ('import', 'vfs', 'file')

    # all done
    return linker


# main
if __name__ == "__main__":
    test()


# end of file
