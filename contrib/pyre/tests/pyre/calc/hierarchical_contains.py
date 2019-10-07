#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that __contains__ is implemented correctly
"""


def test():
    import pyre.calc

    # create a model
    model = pyre.calc.model()

    # the name of the test node
    name = "user.name"
    # register it
    model[name] = "Michael Aïvázis"

    # is the node accessible as a string?
    assert name in model
    # split it and check whether it is accessible as a tuple
    assert tuple(model.split(name)) in model
    # hash it and check whether it is accessible as a key
    assert model.hash(name) in model

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
