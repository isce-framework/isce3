#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the common initialization pattern short-circuits correctly
"""


def foo(arg=None):
    return arg or throw()

def throw():
    raise NotImplementedError("on purpose")

def test():
    # first with an argument -> no exception
    foo(1)

    # now without argument -> exception
    try:
        foo()
    except NotImplementedError:
        pass

    return

# main
if __name__ == "__main__":
    test()


# end of file
