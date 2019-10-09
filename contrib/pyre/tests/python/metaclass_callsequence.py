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


order = []
order.append("declaration: metaclass")

class meta(type):
    order.append("declaration: metaclass __prepare__")
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        order.append("running:  meta.__prepare__")
        return super().__prepare__(name, bases)

    order.append("declaration: metaclass __new__")
    def __new__(metacls, name, bases, attributes, **kwds):
        order.append("running: meta.__new__")
        return super().__new__(metacls, name, bases, attributes)

    order.append("declaration: metaclass __init__")
    def __init__(self, name, bases, attributes, **kwds):
        order.append("running:  meta.__init__")
        super().__init__(name, bases, attributes)
        return

    order.append("declaration: metaclass __call__")
    def __call__(self, **kwds):
        order.append("running: meta.__call__")
        instance = super().__call__(**kwds)
        return instance


order.append("declaration: base")

class base(object, metaclass=meta):

    order.append("declaration: base.__new__")
    def __new__(cls):
        order.append("running:   base.__new__")
        return super().__new__(cls)

    order.append("declaration: base.__init__")
    def __init__(self, **kwds):
        order.append("running:   base.__init__")
        return
    order.append("declaration: end of base")


order.append("declaration: test")
def test():
    order.append("running: instantiating one")
    b1 = base()
    order.append("running: instantiating another")
    b2 = base()

    # now check
    assert order == [
        'declaration: metaclass',
        'declaration: metaclass __prepare__',
        'declaration: metaclass __new__',
        'declaration: metaclass __init__',
        'declaration: metaclass __call__',
        'declaration: base',
        'running:  meta.__prepare__',
        'declaration: base.__new__',
        'declaration: base.__init__',
        'declaration: end of base',
        'running: meta.__new__',
        'running:  meta.__init__',
        'declaration: test',
        'running: test',
        'running: instantiating one',
        'running: meta.__call__',
        'running:   base.__new__',
        'running:   base.__init__',
        'running: instantiating another',
        'running: meta.__call__',
        'running:   base.__new__',
        'running:   base.__init__']

    return


# main
if __name__ == "__main__":
    order.append("running: test")
    test()


# end of file
