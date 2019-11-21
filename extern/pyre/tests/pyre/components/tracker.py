#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
A scratch pad for workflow design
"""

# support
import pyre


# a component
class component(pyre.component, family='pyre.flow.raw'):
    """
    A configurational node: raw meta-data
    """

    # user configurable state
    trait = pyre.properties.float(default=1)


# the driver
def test():
    # instantiate my tracker
    t = pyre.tracker()
    # verify its history is empty
    assert len(t.history) == 0

    # make a component instance
    c = component(name='raw')

    # ask my tracker to wtach it
    t.track(component=c)

    # change the component's trait
    c.trait = 4
    # and again
    c.trait = 16

    # grab the tracker history
    history = t.history
    # the data i am after is stored under the key for {trait}
    records = history[c.pyre_slot('trait').key]

    # the history should be three long: the default plus our two changes
    assert len(records) == 3

    # details: grab the zeroth one
    rev = records[0]
    # check
    assert rev.value == 1
    assert rev.locator.source == './tracker.py'
    assert rev.locator.line == 36
    assert rev.locator.function == 'test'
    assert rev.priority.category == rev.priority.defaults.category

    # grab the first
    rev = records[1]
    # check
    assert rev.value == 4
    assert rev.locator.source == './tracker.py'
    assert rev.locator.line == 42
    assert rev.locator.function == 'test'
    assert rev.priority.category == rev.priority.explicit.category
    assert rev.priority.rank == 0

    # grab the second
    rev = records[2]
    # check
    assert rev.value == 16
    assert rev.locator.source == './tracker.py'
    assert rev.locator.line == 44
    assert rev.locator.function == 'test'
    assert rev.priority.category == rev.priority.explicit.category
    assert rev.priority.rank == 1

    # all done
    return 0


# bootstrap
if __name__ == "__main__":
    # run
    status = test()
    # share
    raise SystemExit(status)


# end of file
