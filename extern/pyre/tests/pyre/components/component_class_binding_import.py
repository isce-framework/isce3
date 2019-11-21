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
    class task(pyre.protocol):
        """a protocol"""
        @pyre.provides
        def do(self):
            """do something"""

    # declare a component
    class worker(pyre.component):
        """a component"""
        job = task(default="import:sample.relax")

    # check that task was bound according to our expectations from sample.py
    assert issubclass(worker.job, pyre.component)
    assert worker.job.__name__ == "relax"
    assert worker.job.pyre_isCompatible(task)

    return worker, task


# main
if __name__ == "__main__":
    test()


# end of file
