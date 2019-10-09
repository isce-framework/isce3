#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that circular dependencies are caught properly
"""


def test():
    import pyre.calc

    # a model
    model = pyre.calc.model()

    # self reference
    try:
        model["cost"] = model.expression("{cost}")
        assert False
    except model.CircularReferenceError:
        pass

    # another model
    model = pyre.calc.model()
    # now validate the graph, expecting the circular reference to raise an exception
    try:
        # a cycle
        model["cost"] = model.expression("{price}")
        model["price"] = model.expression("{cost}")
        assert False
    except model.CircularReferenceError:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
