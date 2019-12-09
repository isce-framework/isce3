# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package provides access to the top level framework manager

The implementation is split into two classes: {Executive} and {Pyre}. The former is a base
class that defines the responsibilities of the executive; the latter the actual instance that
manages the lifetimes of the team of managers necessary to implement all framework
services. All this just in case you want to take ownership of how the framework services are
deployed by creating your own executive instance and registering it in the {pyre} namespace.
"""


# executive factory
def executive(**kwds):
    """
    Factory for the framework executive.

    The pyre executive builds and maintains the top-level objects that manage the various
    framework services
    """
    from .Pyre import Pyre
    return Pyre(**kwds)


# debugging support
_verbose = False
_metaclass_Slot = type

def debug():
    """
    Support for debugging the framework
    """
    # print(" ++ debugging 'pyre.framework'")
    # enable boot-time diagnostics
    global _verbose
    _verbose = True
    # attach {ExtentAware} as the metaclass of {Slot} so we can verify that all instances of
    # this class are properly garbage collected
    from ..patterns.ExtentAware import ExtentAware
    global _metaclass_Slot
    _metaclass_Slot = ExtentAware

    # configure the garbage collector
    import gc
    gc.set_debug(gc.DEBUG_LEAK)
    gc.collect()

    # all done
    return


# end of file
