#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify we can extract all names under a given level in the hierarchy
"""


def test():
    import pyre.calc

    # create a model
    model = pyre.calc.model()

    # register the nodes
    model["pyre.user.name"] = "Michael Aïvázis"
    model["pyre.user.email"] = "michael.aivazis@orthologue.com"
    model["pyre.user.affiliation"] = "orthologue"
    model["pyre.user.signature"] = "{pyre.user.name}+' -- '+{pyre.user.email}"
    model["pyre.user.telephone"] = "+1 626.395.3424"

    # and some aliases
    model.alias(alias="χρήστης", target="pyre.user")
    model.alias(base="χρήστης", alias="όνομα", target="pyre.user.name")

    # here are the canonical names
    names = {
        "pyre.user." + tag
        for tag in ("name", "email", "affiliation", "signature", "telephone") }

    # get all the subnodes of "user"
    target = "pyre.user"
    assert len(names) == len(tuple(model.children(key=target)))
    for key, node in model.children(key=target):
        # check that we got the correct node
        assert model._nodes[key] is node

    # repeat with the alias "χρήστης"
    target = "χρήστης"
    assert len(names) == len(tuple(model.children(key=target)))
    for key, node in model.children(key=target):
        # check that we got the correct node
        assert model._nodes[key] is node

    # visual check
    # model.dump()
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
