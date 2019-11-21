#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check explicit and implicit protocol compatibility
"""


def test():
    import pyre

    # declare a protocol
    class protocol(pyre.protocol):
        """a simple protocol"""
        # properties
        name = pyre.property()
        name.default = "my name"

        @pyre.provides
        def say(self):
            """say my name"""

    # declare a component that claims to implement this protocol explicitly
    class explicit(pyre.component, family="tests.explicit", implements=protocol):
        """a simple component"""
        # properties
        name = pyre.property()
        name.default = "whatever"

        @pyre.export
        def say(self):
            """say my name"""
            return self.name

    # declare a component that implements this protocol implicitly
    class implicit(pyre.component, family="tests.implicit"):
        """a simple component"""
        # properties
        name = pyre.property()
        name.default = "whatever"

        @pyre.export
        def say(self):
            """say my name"""
            return self.name

    # check protocol compatibility
    assert explicit.pyre_isCompatible(protocol)
    assert implicit.pyre_isCompatible(protocol)

    return explicit, implicit, protocol


# main
if __name__ == "__main__":
    test()


# end of file
