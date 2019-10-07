#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify we understand correctly how __mro__ and method chaining interplay
"""


def test():
    class Base(object):

        def m(self, **kwds):
            return ['Base', 'object']

    class A1(Base):

        def m(self, **kwds):
            return ['A1'] + super().m(**kwds)

    class A2(A1):

        def m(self, **kwds):
            return ['A2'] + super().m(**kwds)

    class B1(Base):

        def m(self, **kwds):
            return ['B1'] + super().m(**kwds)

    class B2(B1):

        def m(self, **kwds):
            return ['B2'] + super().m(**kwds)

    class C(B1):

        def m(self, **kwds):
            return ['C'] + super().m(**kwds)

    class D(A2, B2, C):

        def m(self, **kwds):
            return ['D'] + super().m(**kwds)

    d = D()
    mro = [ kls.__name__ for kls in type(d).__mro__ ]
    assert d.m() == mro

    return


# main
if __name__ == "__main__":
    test()


# end of file
