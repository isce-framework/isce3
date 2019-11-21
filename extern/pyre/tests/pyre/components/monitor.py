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


# my monitor
class monitor(pyre.monitor):
    """
    A simple monitor that just count the number of times it got a change event
    """

    # hook
    def flush(self, observable, **kwds):
        """
        Handler of change events
        """
        # update my counter
        self.counter += 1
        # chain up
        return super().flush(observable=observable, **kwds)

    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my counter
        self.counter = 0
        # all done
        return


# a component
class component(pyre.component, family='pyre.flow.raw'):
    """
    A configurational node: raw meta-data
    """

    # user configurable state
    trait = pyre.properties.float(default=1)


# the driver
def test():
    # instantiate my monitor
    m = monitor()
    # verify its counter is at zero
    assert m.counter == 0

    # make a component instance
    c = component(name='raw')

    # ask my monitor to wtach it
    m.watch(component=c)

    # change the component's trait
    c.trait = 4
    # verify the monitor counter is now one
    assert m.counter == 1

    # and again
    c.trait = 16
    # verify the monitor counter is now two
    assert m.counter == 2

    # all done
    return 0


# bootstrap
if __name__ == "__main__":
    # run
    status = test()
    # share
    raise SystemExit(status)


# end of file
