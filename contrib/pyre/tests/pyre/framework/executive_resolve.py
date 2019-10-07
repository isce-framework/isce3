#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the executive can retrieve components
"""


def test():
    # framework
    import pyre
    # and its parts
    executive = pyre.executive
    fileserver = executive.fileserver

    # retrieve a component descriptor from the python path
    bases = tuple(executive.resolve(uri="import:pyre.component"))
    for base in bases: assert base is pyre.component
    # retrieve a component descriptor from a file using the virtual filesystem
    d1, *_ = executive.resolve(uri="vfs:{}/sample.py/d1".format(fileserver.STARTUP_DIR))
    # check that one derives from the other
    assert issubclass(d1, base)
    # retrieve a component descriptor from a file using the physical filesystem
    d2,  *_= executive.resolve(uri="file:sample.py/d2")
    # check that one derives from the other
    assert issubclass(d2, base)

    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
