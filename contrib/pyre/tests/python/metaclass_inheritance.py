#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


"""
When a metaclass understands the extra keywords that can be passed during class declaration,
it has to override all these to accommodate the change in signature
"""


class meta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        assert metacls.__name__ == 'meta'
        assert name in ['base', 'derived']
        if name == 'base':
            assert bases == (object,)
            assert kwds == {'arg1': True, 'arg2': False}
        if name == 'derived':
            assert bases == (base,)
            assert kwds == {'arg1': False, 'arg2': True}
        return super().__prepare__(name, bases)


    def __new__(metacls, name, bases, attributes, **kwds):
        assert metacls.__name__ == 'meta'
        assert name in ['base', 'derived']
        if name == 'base':
            assert bases == (object,)
            assert kwds == {'arg1': True, 'arg2': False}
        if name == 'derived':
            assert bases == (base,)
            assert kwds == {'arg1': False, 'arg2': True}
        return super().__new__(metacls, name, bases, attributes)


    def __init__(self, name, bases, attributes, **kwds):
        assert self.__name__ in ['base', 'derived']
        if self.__name__ == 'base':
            assert bases == (object,)
            assert kwds == {'arg1': True, 'arg2': False}
        if self.__name__ == 'derived':
            assert bases == (base,)
            assert kwds == {'arg1': False, 'arg2': True}
        super().__init__(name, bases, attributes)
        return


class base(object, metaclass=meta, arg1=True, arg2=False):


    def __init__(self, **kwds):
        assert type(self).__name__ == 'base'
        assert kwds == {}
        return


class derived(base, arg1=False, arg2=True):


    def __init__(self, **kwds):
        assert type(self).__name__ == 'derived'
        assert kwds == {}
        return


def test():
    b = base()
    d = derived()
    return


# main
if __name__ == "__main__":
    test()


# end of file
