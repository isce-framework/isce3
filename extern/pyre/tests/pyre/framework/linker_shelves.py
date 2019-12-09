#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify the linker can understand shelf uris
"""


def test():
    import pyre
    # build the executive
    executive = pyre.executive
    # access the linker
    linker = executive.linker
    # and the fileserver
    fs = executive.fileserver

    # request a non-existent python module
    try:
        shelf = linker.loadShelf(executive=executive, uri="import:nomodule.nosymbol")
        assert False
    except linker.FrameworkError as error:
        pass

    # request a file
    shelf = linker.loadShelf(executive=executive, uri="file:sample.py")
    # make sure it got imported correctly
    assert shelf.retrieveSymbol('factory')

    # request a non-existent file
    try:
        shelf = linker.loadShelf(executive=executive, uri="file:not-there.py")
        assert False
    except linker.FrameworkError as error:
        pass

    # request a file with a syntax error
    try:
        shelf = linker.loadShelf(executive=executive, uri="file:sample_syntaxerror.py")
        assert False
    except SyntaxError as error:
        pass

    # request the same file through vfs
    shelf = linker.loadShelf(executive=executive, uri="vfs:{}/sample.py".format(fs.STARTUP_DIR))
    # make sure it got imported correctly
    assert shelf.retrieveSymbol('factory')

    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
