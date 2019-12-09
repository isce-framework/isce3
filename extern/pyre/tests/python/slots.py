#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the __slots__ mechanism works as expected
"""


class base(object):
    classvar = 0

    def __init_(self, **kwds):
        super().__init__(**kwds)
        self.instancevar = 0

    __slots__ = ("instancevar", )


class derived(base):

    def __init__(self, **kwds):
        super().__init__(**kwds)
        return

    __slots__ = ()


def test():
    assert base.__slots__ == ("instancevar", )
    b = base()
    b.instancevar = 'b'

    try:
        b.classvar = 'b'
    except AttributeError as error:
        assert str(error) == "'base' object attribute 'classvar' is read-only"

    try:
        b.foo = 5
        assert False
    except AttributeError as error:
        assert str(error) == "'base' object has no attribute 'foo'"

    d = derived()
    d.instancevar = 'd'
    try:
        d.foo = 5
        assert False
    except AttributeError as error:
        assert str(error) == "'derived' object has no attribute 'foo'"

    return


# main
if __name__ == "__main__":
    test()


# end of file
