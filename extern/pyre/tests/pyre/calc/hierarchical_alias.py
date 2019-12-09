#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the aliasing feature of hierarchical models
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

    # case 1: canonical does not exist, alias does not exist
    model.alias(base="author", alias="affiliation", target="user.affiliation")
    model["user.affiliation"] = "orthologue"
    assert model["author.affiliation"] == model["user.affiliation"]

    # case 2: canonical exists, alias does not
    model.alias(base="author", alias="signature", target="user.signature")
    # check the signature
    assert model["author.signature"] == model["user.signature"]

    # case 3: canonical does not exist, alias does
    model["author.telephone"] = "+1 626.395.3424"
    model.alias(base="author", alias="telephone", target="user.telephone")
    assert model["author.telephone"] == model["user.telephone"]

    # case 4: both are preëxisting nodes
    model["author.name"] = "TBD"
    try:
        model.alias(base="author", alias="name", target="user.name")
        assert False
    except model.AliasingError as error:
        pass

    # check the final state
    assert model["user.name"] == "Michael Aïvázis"
    assert model["user.email"] == "michael.aivazis@orthologue.com"
    assert model["user.telephone"] == "+1 626.395.3424"
    assert model["user.affiliation"] == "orthologue"
    assert model["user.signature"] == "Michael Aïvázis -- michael.aivazis@orthologue.com"

    # and again through the aliases
    assert model["author.name"] == "Michael Aïvázis"
    try:
        model["author.email"]
        assert False
    except model.UnresolvedNodeError as error:
        pass
    assert model["author.telephone"] == "+1 626.395.3424"
    assert model["author.affiliation"] == "orthologue"
    assert model["author.signature"] == "Michael Aïvázis -- michael.aivazis@orthologue.com"

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
