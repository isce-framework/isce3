#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify processing of a correct pfg input file
"""


def test():
    # package access
    import pyre.config
    from pyre.config.events import ConditionalAssignment
    # get the codec manager
    m = pyre.config.newConfigurator()
    # ask for a pfg codec
    reader = m.codec(encoding="pfg")
    # the configuration file
    uri = "sample-componentConditionalNested.pfg"
    # open a stream
    sample = open(uri)
    # read the contents
    events = tuple(reader.decode(uri=uri, source=sample, locator=None))
    # check that we got a non-trivial instance
    assert events

    # verify its contents
    event = events[0]
    assert isinstance(event, ConditionalAssignment)
    assert event.component == ["sample", "engine"]
    assert event.conditions == [
        (['sample', 'engine'], ['test', 'part']),
        (['sample'], ['test', 'item'])
        ]
    assert event.key == ["id"]
    assert event.value == '3Q4XYZ'

    return m, reader, events


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
