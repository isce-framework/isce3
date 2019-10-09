#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Investigate __format__
"""


class sample(object):
    """test class that overloads __format__"""

    def __init__(self, value):
        self.value = value

    def __format__(self, code):
        return self.value.__format__(code)

    def __repr__(self):
        return "repr"

    def __str__(self):
        return "str"


def test():
    s = sample(3.14159)

    # the type coercions
    assert "{0!a}".format(s) == "repr"
    assert "{0!r}".format(s) == "repr"
    assert "{0!s}".format(s) == "str"

    # the format specification
    assert "3.14159" == "{}".format(s)
    assert "3.14" == "{:.2f}".format(s)

    return


# main
if __name__ == "__main__":
    test()


# end of file
