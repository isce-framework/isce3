#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

"""
Verify that the class decorator works as expected
"""


def test():
    # get the package parts
    from pyre.schemata import typed, list, bool

    # make a base class
    class node: pass

    # use the decorator with no arguments to derive a typed subclass
    @typed
    class t1(node): pass
    # use it to make a complex schema: a list of booleans
    b = t1.list(schema=t1.bool())
    # check it
    assert b.coerce('y,n,yes,no,on,off') == [True, False, True, False, True, False]

    # use the decorator with default arguments to do the same
    @typed()
    class t2(node): pass
    # use it to make a complex schema: a list of booleans
    b = t2.list(schema=t2.bool())
    # check it
    assert b.coerce('y,n,yes,no,on,off') == [True, False, True, False, True, False]

    # restrict the supported types to a specific list
    @typed(schemata=[list, bool])
    class t3(node): pass
    # use it to make a complex schema: a list of booleans
    b = t3.list(schema=t3.bool())
    # check it
    assert b.coerce('y,n,yes,no,on,off') == [True, False, True, False, True, False]
    # check that it doesn't have the other types
    try:
        # by accessing one of them
        t3.int
        assert False
    # if it complains
    except AttributeError:
        # well done
        pass

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
