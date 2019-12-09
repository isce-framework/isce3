#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that facilities get bound correctly when specified implicitly
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
        task = job(default="import:sample.relax")

    # instantiate
    c = component(name="c")
    # check
    assert isinstance(c.task, pyre.component)
    assert c.task.pyre_name == "c.task"

    return c, component, job


# main
if __name__ == "__main__":
    test()


# end of file
