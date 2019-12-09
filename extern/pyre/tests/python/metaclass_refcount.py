#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


"""
Explore an implementation strategy for making classes aware of their extent
"""


import weakref


class ExtentManaged(type):


    def __new__(cls, name, bases, attributes):
        # buld the record
        record = super().__new__(cls, name, bases, attributes)

        # add the weakset attribute
        record._extent = weakref.WeakSet()

        # grab the constructor; it is guaranteed to exist since object has one
        constructor = record.__init__
        # declare the replacement
        def _constructor(self, **kwds):
            constructor(self, **kwds)
            record._extent.add(self)
            return

        record.__init__ = _constructor

        return record


class Base(object, metaclass=ExtentManaged):
    """base class"""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        return


class Derived(Base):
    """derived class"""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        return


def make():
    b1 = Base()
    b2 = Base()
    d1 = Derived()
    d2 = Derived()

    assert set(Base._extent) == { b1, b2, d1, d2 }
    assert set(Derived._extent) == { d1, d2 }

    return

def test():
    make()

    assert set(Base._extent) == set()
    assert set(Derived._extent) == set()

    return


# main
if __name__ == "__main__":
    test()


# end of file
