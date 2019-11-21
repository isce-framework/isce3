#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we understand attribute storage in the presence of a custom container
"""


class mydict(dict):

    def __setitem__(self, key, value):
        # print("setting {0!r} -> {1!r}".format(key, value))
        return super().__setitem__(key, value)


class meta(type):
    # print("declaring the metaclassclass __prepare__")
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        # print("preparing custom storage")
        return mydict()


# print("declaring the class")

class base(object, metaclass=meta):
    """a class"""

    clsvar = True

    def __init__(self):
        self.instancevar = True
        return


def test():

    # the magic ones
    assert hasattr(base, '__module__')
    assert hasattr(base, '__doc__')
    assert hasattr(base, '__dict__')
    assert hasattr(base, '__weakref__')

    # the ones i declared
    assert hasattr(base, '__init__')
    assert hasattr(base, 'clsvar')

    # check that clsvar is true
    assert base.clsvar == True

    # instantiate one
    b = base()

    assert hasattr(b, '__module__')
    assert hasattr(b, '__doc__')
    assert hasattr(b, '__dict__')
    assert hasattr(b, '__weakref__')

    # the ones i declared
    assert hasattr(b, '__init__')
    assert hasattr(b, 'clsvar')
    assert hasattr(b, 'instancevar')

    # check that clsvar is true
    assert b.clsvar == True
    assert b.instancevar == True

    return


# main
if __name__ == "__main__":
    test()


# end of file
