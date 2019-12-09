#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
When a metaclass understands the extra keywords that can be passed during class declaration,
it has to override all these to accommodate the change in signature
"""


# print("declaring the metaclass")

class meta(type):
    # print("declaring the metaclassclass __prepare__")
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        assert metacls.__name__ == 'meta'
        assert name == 'base'
        assert bases == (object,)
        assert kwds == {'arg1': True, 'arg2': False}

        # print("{0.__name__!r}.__prepare__".format(metacls))

        return super().__prepare__(name, bases)



    # print("declaring the metaclassclass __new__")
    def __new__(metacls, name, bases, attributes, **kwds):
        assert metacls.__name__ == 'meta'
        assert name == 'base'
        assert bases == (object,)
        assert kwds == {'arg1': True, 'arg2': False}

        # print("{0.__name__!r}.__new__".format(metacls))

        return super().__new__(metacls, name, bases, attributes)


    # print("declaring the metaclassclass constructor")
    def __init__(self, name, bases, attributes, **kwds):
        assert self.__name__ == 'base'
        assert name == 'base'
        assert bases == (object,)
        assert kwds == {'arg1': True, 'arg2': False}

        # print(type(self))
        # print("{0.__name__!r}.__init__".format(self))
        # print(dir(self))

        super().__init__(name, bases, attributes)
        return


# print("declaring the class")

class base(object, metaclass=meta, arg1=True, arg2=False):


    # print("declaring the class constructor")
    def __init__(self, **kwds):
        # print("{.__name__!r}.__init__".format(type(self)))
        # print(dir(self))
        assert type(self).__name__ == 'base'
        assert kwds == {}
        return
    # print("done declaring the class constructor")


def test():
    b1 = base()
    b2 = base()
    return


# main
if __name__ == "__main__":
    test()


# end of file
