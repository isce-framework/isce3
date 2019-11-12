#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that properties are accessible from derived classes
"""


class Base1(object):

    @property
    def name(self): return "base1"


class Derived(Base1):
    pass


def test():
    d = Derived()
    assert d.name == "base1"
    return


# main
if __name__ == "__main__":
    test()


# end of file
