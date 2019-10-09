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

class here(Exception):
    """signal that a particular line of code was reached"""
    def __init__(self, loc):
        self.loc = loc
    def __str__(self):
        return self.loc

class meta(type):
    """""the metaclass that traps class attribute assignment"""
    def __setattr__(self, name, value):
        raise here(loc=value+"."+name)

class base(object, metaclass=meta):
    """a base class"""

class derived(base):
    """a derived class"""


def test():
    # check attribute assignment in the base class
    try:
        base.foo = "base"
    except here as marker:
        assert str(marker) == "base.foo"

    # check attribute assignment in the derived class
    try:
        derived.foo = "derived"
    except here as marker:
        assert str(marker) == "derived.foo"

    # verify instances are unaffected
    # the base
    b = base()
    b.foo = "b.foo"
    assert b.foo == "b.foo"
    # the derived
    d = derived()
    d.foo = "d.foo"
    assert d.foo == "d.foo"

    return


# main
if __name__ == "__main__":
    test()


# end of file
