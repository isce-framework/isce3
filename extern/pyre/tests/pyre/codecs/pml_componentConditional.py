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
    uri = "sample-componentConditional.pml"
    # open a stream
    sample = open(uri)
    # read the contents
    events = reader.decode(uri=uri, source=sample, locator=None)
    # check that we got a non-trivial instance
    assert events

    # verify its contents
    event = events[0]
    assert isinstance(event, ConditionalAssignment)
    assert event.component == ["mine"]
    assert event.conditions == [(["mine"], ["test", "sample"])]
    assert event.key == ["author"]
    assert event.value == "Michael Aïvázis"

    event = events[1]
    assert isinstance(event, ConditionalAssignment)
    assert event.component == ["mine"]
    assert event.conditions == [(["mine"], ["test", "sample"])]
    assert event.key == ["affiliation"]
    assert event.value == "orthologue"

    return m, reader, events


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
