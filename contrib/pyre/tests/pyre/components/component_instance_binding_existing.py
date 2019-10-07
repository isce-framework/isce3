#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that facilities get bound correctly to existing instances
"""


def test():
    import pyre

    # declare a protocol
    class job(pyre.protocol):
        """a protocol"""
        @pyre.provides
        def do(self):
            """do something"""

    # declare a component
    class component(pyre.component):
        """a component"""
        w1 = job()
        w2 = job()

    # instantiate
    c = component(name="c")
    # bind {w1} and {w2}
    c.w1 = "import:sample.relax#worker"
    c.w2 = "import:sample.relax#worker"
    # check
    assert c.w1 == c.w2
    assert c.w1.pyre_name == 'worker'

    return c, component, job


# main
if __name__ == "__main__":
    test()


# end of file
