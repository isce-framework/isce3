#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that trait defaults get bound correctly
"""


def test():
    import pyre

    # declare a protocol
    class task(pyre.protocol):
        """a protocol"""
        @pyre.provides
        def do(self):
            """do something"""

        @classmethod
        def pyre_default(cls, **kwds):
            """the default task"""
            return relax

    # declare a component that implements this protocol
    class relax(pyre.component, implements=task):
        """an implementation"""
        @pyre.export
        def do(self):
            """do nothing"""

    # declare a component
    class worker(pyre.component):
        """the base component"""
        uid = pyre.properties.int(default=1)
        duties = task()

    # check the default values
    assert worker.uid == 1
    assert worker.duties == relax

    return worker


# main
if __name__ == "__main__":
    test()


# end of file
