# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for all the exceptions raised by this package
"""


from ..framework.exceptions import FrameworkError


class NodeError(FrameworkError):
    """
    Base class for pyre.algebraic errors. Useful as a catch-all
    """


class CircularReferenceError(NodeError):
    """
    Signal a circular reference in the evaluation graph
    """

    # public data
    description = "the evaluation graph has a cycle at {0.node}"

    # meta-methods
    def __init__(self, node, path=(), **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.node = node
        self.path = path
        # all done
        return


# end of file
