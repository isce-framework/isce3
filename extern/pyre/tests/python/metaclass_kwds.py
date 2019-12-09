#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


"""
Verify that a metaclass that understands extra keywords that can be passed during class
declaration has to override all these to accommodate the change in signature
"""


# print("declaring the metaclass")

class meta(type):
    # print("  declaring the metaclassclass __prepare__")
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        # print("  meta.__prepare__")
        return super().__prepare__(name, bases)



    # print("  declaring the metaclassclass __new__")
    def __new__(metacls, name, bases, attributes, **kwds):
        # print("  meta.__new__")
        # print("    metacls: {0!r}".format(metacls))
        # print("    name: {0!r}".format(name))
        # print("    bases: {0!r}".format(bases))
        # print("    attributes: {0!r}".format(attributes))
        # print("    kwds: {0!r}".format(kwds))
        record =  super().__new__(metacls, name, bases, attributes)
        # print("  meta.__new__: record:")
        # print("    record: {0!r}".format(record))
        # print("      record.__dict__: {0!r}".format(record.__dict__.keys()))

        return record


    # print("  declaring the metaclassclass __init__")
    def __init__(self, name, bases, attributes, arg1, arg2):
        """
        this gets called after the class record has been built by type
        it is passed in as self
        we can now add class variables to the record
        """
        super().__init__(name, bases, attributes)
        # print("  meta.__init__:")
        # print("    self: {0!r}".format(self))
        # print("      self.__dict__: {0!r}".format(self.__dict__.keys()))
        # print("    name: {0!r}".format(name))
        # print("    bases: {0!r}".format(bases))
        # print("    attributes: {0!r}".format(attributes))
        # print("    kwds: {0!r}".format(kwds))

        self.arg1 = arg1
        self.arg2 = arg2

        def foo(arg): print(arg)

        self.foo = foo

        return


# print("declaring the class")

class base(object, metaclass=meta, arg1=True, arg2=False):


    # print("    declaring the base __init__")
    def __init__(self, **kwds):
        # print("base.__init__")
        super().__init__(**kwds)
        return


def test():
    """
    Verify that a metaclass that understands extra keywords that can be passed during class
    declaration has to override all these to accommodate the change in signature
    """

    # check the class record
    assert base.arg1 == True
    assert base.arg2 == False
    assert base.foo

    # check an instance
    b = base()
    assert b.arg1 == True
    assert b.arg2 == False
    assert b.foo
    # print(b.foo)
    # b.foo()
    return


# main
if __name__ == "__main__":
    test()


# end of file
