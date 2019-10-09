#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that expressions work
"""


def test():
    import pyre.calc

    # set up the model
    model = pyre.calc.model()

    # the nodes
    p = 80.
    s = .25*80
    # register the nodes
    model["production"] = p
    model["shipping"] = s
    model["cost"] = model.expression("{production}+{shipping}")
    model["price"] = model.expression("2*{cost}")

    # check the values
    # print("before:")
    # print("  production:", model["production"])
    # print("  shipping:", model["shipping"])
    # print("  cost:", model["cost"])
    # print("  price:", model["price"])
    assert model["production"] == p
    assert model["shipping"] == s
    assert model["cost"] == p+s
    assert model["price"] == 2*(p+s)

    # thanks to the indirect references to their operands, expressions get updated values when
    # their dependents change; this is a side effect of the current implementation

    # make a change
    p = 100.
    model["production"] = p

    # check again
    # print("after:")
    # print("  production:", model["production"])
    # print("  shipping:", model["shipping"])
    # print("  cost:", model["cost"])
    # print("  price:", model["price"])
    assert model["production"] == p
    assert model["shipping"] == s
    assert model["cost"] == p+s
    assert model["price"] == 2*(p+s)

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
