#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
How do you endow classes and their instances with the same interface?
"""


# print("declaring the metaclass")
class meta(type):
    def metafoo(cls): return "meta"

# print("declaring the class")
class base(metaclass=meta):
    @classmethod
    def classfoo(cls): return "class"
    def foo(self): return "instance"


def test():
    # first the class
    try:
        base.foo()
        assert False
    except TypeError:
        pass

    assert base.metafoo() == "meta"
    assert base.classfoo() == "class"

    # now, an instance
    b = base()
    assert b.foo() == "instance"
    assert b.classfoo() == "class"
    try:
        b.metafoo()
        assert False
    except AttributeError:
        pass

    # print(b.foo)
    # print(b.foo.__self__)
    # print(b.foo.__func__)

    # print(base.metafoo)
    # print(base.classfoo)

    return


# main
if __name__ == "__main__":
    test()


# end of file
