#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we understand how multiple inheritance shadows duplicate attribute names
"""


class Base1(object):
    name = "base1"

class Base2(object):
    name = "base2"

class Derived(Base1, Base2):
    pass


def test():
    d = Derived()
    assert d.name == "base1"
    return


# main
if __name__ == "__main__":
    test()


# end of file
