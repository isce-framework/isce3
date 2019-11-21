#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the path primitive
"""


def test():
    # the home of the factory
    import pyre.primitives

    # make one
    root = pyre.primitives.path('/')
    # check
    assert str(root) == '/'

    # make another
    here = pyre.primitives.path('pyre-1.0/tests/pyre/primitives/path.py')
    # check
    assert str(here) == 'pyre-1.0/tests/pyre/primitives/path.py'

    # make another
    total = pyre.primitives.path(root, 'Users/mga/dv', here)
    # check
    assert str(total) == '/Users/mga/dv/pyre-1.0/tests/pyre/primitives/path.py'

    # check that construction fails when handed the wrong argument type
    try:
        # by making a bad one
        pyre.primitives.path(1)
        # and making sure we don't get here
        assert False
    # if it fails, as it should
    except TypeError as error:
        # show me
        # print(error)
        # and check
        assert str(error) == "can't parse '1', of type <class 'int'>"

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
