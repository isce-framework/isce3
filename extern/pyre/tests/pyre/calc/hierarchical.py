#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: instantiate a hierarchical model
"""


def test():
    import pyre.calc

    # create a model
    model = pyre.calc.model()

    # register the nodes
    model["user.name"] = "Michael Aïvázis"
    model["user.email"] = "michael.aivazis@orthologue.com"
    model["user.signature"] = model.interpolation("{user.name} -- {user.email}")
    # check the signature
    assert model["user.signature"] == "Michael Aïvázis -- michael.aivazis@orthologue.com"

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
