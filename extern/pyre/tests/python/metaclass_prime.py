#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we can inject symbols in the context of the class
"""


class mydict(dict):

    @staticmethod
    def decorate(arg):
        arg.decorated = True
        return arg

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self["decorate"] = self.decorate


class meta(type):
    # print("declaring the metaclassclass __prepare__")
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        # print("preparing custom storage")
        return mydict()


class base(object, metaclass=meta):
    """a class"""

    @decorate
    def do(self):
        """a method"""


def test():
    assert base.do.decorated == True
    return


# main
if __name__ == "__main__":
    test()


# end of file
