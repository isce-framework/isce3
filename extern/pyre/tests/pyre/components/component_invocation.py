#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that component behaviors are callable
"""


def test():
    import pyre

    # declare a component
    class component(pyre.component):
        """a test component"""
        # behavior
        @pyre.export
        def do(self):
            """behave"""
            return True

    # instantiate it
    c = component(name="test")
    # invoke its behavior
    assert c.do()
    # and return it
    return c


# main
if __name__ == "__main__":
    test()


# end of file
