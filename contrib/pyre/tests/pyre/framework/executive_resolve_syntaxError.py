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
    executive =  pyre.executive

    # retrieve a component descriptor from the python path
    try:
        a, = executive.resolve(uri="file:sample_syntaxerror.py/factory")
        assert False
    except SyntaxError as error:
        pass
    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
