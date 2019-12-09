# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref


# class declaration
class Simplex:
    """
    A representation of a simplex of arbitrary order
    """


    # public data
    @property
    def order(self):
        """
        Compute my order
        """
        # easy enough
        return len(self.support) - 1


    # meta-methods
    def __init__(self, support=None, **kwds):
        # chain up
        super().__init__(**kwds)

        # the set of higher order simplices i support
        self.wings = weakref.WeakSet()

        # if i don't have a non-trivial support
        if support is None:
            # mark me as a simplex of order 0; this marker is meant to make {order} work
            self.support = self.zero
        # otherwise
        else:
            # save my support
            self.support = tuple(support)
            # and visit each simplex in it
            for simplex in self.support:
                # to add me to its wings
                simplex.wings.add(self)

        # all done
        return


    # implementation details
    zero = (None,)


# end of file
