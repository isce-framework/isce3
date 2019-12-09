#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise setting configuration values through the interface used during configuration event
processing
"""


def test():
    # access the framework
    import pyre
    # and its managers
    executive = pyre.executive
    ns = pyre.executive.nameserver
    cfg = pyre.executive.configurator

    # the priority category for this test
    priority = executive.priority.user

    # make some events
    events = [
        cfg.events.Assignment(
            key=("sample", "user", "name"), value="michael aïvázis",
            locator=pyre.tracking.here()),
        cfg.events.Assignment(
            key=("sample", "user", "affiliation"), value="orthologue",
            locator=pyre.tracking.here()),
        cfg.events.Assignment(
            key=("sample", "user", "email"), value="michael.aivazis@orthologue.com",
            locator=pyre.tracking.here()),
        cfg.events.Assignment(
            key=("sample", "user", "alias"), value="{sample.user.name}",
            locator=pyre.tracking.here())
        ]

    # process them
    cfg.processEvents(events=events, priority=priority)

    # dump the contents of the model
    # ns.dump()

    # check the variable bindings
    assert ns["sample.user.name"] == "michael aïvázis"
    assert ns["sample.user.email"] == "michael.aivazis@orthologue.com"
    assert ns["sample.user.affiliation"] == "orthologue"
    assert ns["sample.user.alias"] == ns["sample.user.name"]

    # and return the manager
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
