#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify processing of a correct pml input file
"""


def test():
    # package access
    import pyre.config
    from pyre.config.events import Assignment, ConditionalAssignment
    # get the codec manager
    m = pyre.config.newConfigurator()
    # ask for a pml codec
    reader = m.codec(encoding="pml")
    # the configuration file
    uri = "sample.pml"
    # open a stream
    sample = open(uri)
    # read the contents
    events = reader.decode(uri=uri, source=sample, locator=None)
    # check that we got a non-trivial instance
    assert events

    # verify its contents
    event = events[0]
    assert isinstance(event, Assignment)
    assert event.key == ["gauss", "name"]
    assert event.value == "gauss"


    event = events[1]
    assert isinstance(event, Assignment)
    assert event.key == ["gauss", "mc", "samples"]
    assert event.value == "10**6"

    event = events[2]
    assert isinstance(event, Assignment)
    assert event.key == ["gauss", "mc", "integrand"]
    assert event.value == "import:gauss.fuctors.gaussian"

    event = events[3]
    assert isinstance(event, Assignment)
    assert event.key == ["gauss", "mc", "box", "diagonal"]
    assert event.value == "((-1,-1), (1,1))"

    event = events[4]
    assert isinstance(event, ConditionalAssignment)
    assert event.component == ["gauss", "mc", "integrand"]
    assert event.conditions == [
        (["gauss", "mc", "integrand"], ["gauss", "functors", "gaussian"])
        ]
    assert event.key == ["μ"]
    assert event.value == "(0,0)"

    event = events[5]
    assert isinstance(event, ConditionalAssignment)
    assert event.component == ["gauss", "mc", "integrand"]
    assert event.conditions == [
        (["gauss", "mc", "integrand"], ["gauss", "functors", "gaussian"])
        ]
    assert event.key == ["σ"]
    assert event.value == "1/3"

    return m, reader, events


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
