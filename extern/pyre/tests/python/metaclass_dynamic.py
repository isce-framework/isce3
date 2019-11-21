#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we can specify the metaclass dynamically and the classes still get built properly
"""


class aspectOne(type): pass
class aspectTwo(type): pass


def classFactory(metaclass=type):
    """
    build a class with the given metaclass
    """
    class base(object, metaclass=metaclass): pass

    return base


def test():

    zero = classFactory()
    assert type(zero) == type

    one = classFactory(metaclass=aspectOne)
    assert type(one) == aspectOne

    two = classFactory(metaclass=aspectTwo)
    assert type(two) == aspectTwo

    return


# main
if __name__ == "__main__":
    test()


# end of file
