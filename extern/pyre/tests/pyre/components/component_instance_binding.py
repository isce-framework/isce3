#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the trait defaults get bound correctly
"""


def test():
    import pyre

    # declare a protocol
    class job(pyre.protocol):
        """a protocol"""
        @pyre.provides
        def do(self):
            """do something"""

    # declare a component the implements this protocol
    class worker(pyre.component, implements=job):
        """an implementation"""
        @pyre.export
        def do(self):
            """do something"""

    # declare a component
    class base(pyre.component):
        """the base component"""
        number = pyre.properties.int(default=1)
        task = job(default=worker)
        @pyre.export
        def say(self):
            """say something"""

    class derived(base):
        """the derived component"""
        length = pyre.properties.float(default=10.)

    # instantiate
    d = derived(name="d")
    # check the inventory
    assert d.number == 1
    assert d.length == 10
    assert isinstance(d.task, worker)

    return base


# main
if __name__ == "__main__":
    test()


# end of file
