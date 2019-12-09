#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the codec manager can be instantiated
"""


def test():
    import pyre
    # build the executive
    executive = pyre.executive

    # request a python module
    shelf = executive.loadShelf(uri="import:math")
    # make sure it got imported correctly
    assert shelf.retrieveSymbol('pi')

    # request a non-existent python module
    try:
        shelf = executive.loadShelf(uri="import:foo")
        assert False
    except executive.FrameworkError as error:
        pass

    # request a non-existent python module
    try:
        shelf = executive.loadShelf(uri="import:nomodule.nosymbol")
        assert False
    except executive.FrameworkError as error:
        pass

    # request a file
    shelf = executive.loadShelf(uri="file:sample.py")
    # make sure it got imported correctly
    assert shelf.retrieveSymbol('factory')

    # request a non-existent file
    try:
        shelf = executive.loadShelf(uri="file:not-there.py")
        assert False
    except executive.FrameworkError as error:
        pass

    # request a file with a syntax error
    try:
        shelf = executive.loadShelf(uri="file:sample_syntaxerror.py")
        assert False
    except SyntaxError as error:
        pass

    # request the same file through vfs
    shelf = executive.loadShelf(uri="vfs:/local/sample.py")
    # make sure it got imported correctly
    assert shelf.retrieveSymbol('factory')


    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
