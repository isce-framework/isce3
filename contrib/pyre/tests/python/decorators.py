#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that function decorators leave the name of the function unchanged
"""


def test():
    def decorator(func): return func

    class base(object):

        @decorator
        def f(self): return

    b = base()
    b.f()

    return base, decorator


# main
if __name__ == "__main__":
    test()


# end of file
