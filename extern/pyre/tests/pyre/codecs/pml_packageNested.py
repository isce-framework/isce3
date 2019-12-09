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
    from pyre.config.events import Assignment
    # get the codec manager
    m = pyre.config.newConfigurator()
    # ask for a pml codec
    reader = m.codec(encoding="pml")
    # the configuration file
    uri = "sample-packageNested.pml"
    # open a stream
    sample = open(uri)
    # read the contents
    events = reader.decode(uri=uri, source=sample, locator=None)
    # check that we got a non-trivial instance
    assert events

    # verify its contents
    event = events[0]
    assert isinstance(event, Assignment)
    assert tuple(event.key) == ("pyre", "home")
    assert event.value == "pyre.home()"

    event = events[1]
    assert isinstance(event, Assignment)
    assert tuple(event.key) == ("pyre", "prefix")
    assert event.value == "pyre.prefix()"

    event = events[2]
    assert isinstance(event, Assignment)
    assert tuple(event.key) == ("pyre", "user", "name")
    assert event.value == "michael a.g. aïvázis"

    event = events[3]
    assert isinstance(event, Assignment)
    assert tuple(event.key) == ("pyre", "user", "email")
    assert event.value == "michael.aivazis@orthologue.com"

    event = events[4]
    assert isinstance(event, Assignment)
    assert tuple(event.key) == ("pyre", "user", "affiliation")
    assert event.value == "orthologue"

    return m, reader, events


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
