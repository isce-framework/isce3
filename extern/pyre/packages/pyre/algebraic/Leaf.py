# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Leaf:
    """
    Mix-in class that provides an implementation of the subset of the interface of {Node} that
    requires traversals of the expression graph rooted at leaf nodes.
    """


    # interface
    @property
    def span(self):
        """
        Traverse my subgraph and yield all its nodes
        """
        # just myself
        yield self
        # and nothing else
        return


# end of file
