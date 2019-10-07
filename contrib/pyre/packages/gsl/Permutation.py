# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import numbers
from . import gsl # the extension


# the class declaration
class Permutation:
    """
    A wrapper over a gsl permutation
    """

    # initialization
    def init(self):
        """
        Initialize a permutation
        """
        # initialize me
        gsl.permutation_init(self.data)
        # and return
        return self


    def clone(self):
        """
        Allocate a new permutation and initialize it using my values
        """
        # build the clone
        clone = type(self)(shape=self.shape)
        # have the extension initialize the clone
        gsl.permutation_copy(clone.data, self.data)
        # and return it
        return clone


    def reverse(self):
        """
        Reverse me
        """
        # reverse
        gsl.permutation_reverse(self.data)
        # and return
        return self


    def inverse(self):
        """
        Reverse me
        """
        # inverse
        gsl.permutation_invverse(self.data)
        # and return
        return self


    def swap(self, other):
        """
        Swap me with {other}
        """
        # perform the swap
        gsl.permutation_swap(self.data, other.data)
        # and return me
        return self


    def size(self):
        """
        Compute my size
        """
        # easy enough
        return gsl.permutation_size(self.data)


    def next(self):
        """
        Compute the next permutation in my sequence
        """
        # easy enough
        return gsl.permutation_next(self.data)


    def prev(self):
        """
        Compute the prev permutation in my sequence
        """
        # easy enough
        return gsl.permutation_prev(self.data)


    # meta methods
    def __init__(self, shape, data=None, **kwds):
        super().__init__(**kwds)
        self.shape = shape
        self.data = gsl.permutation_alloc(shape) if data is None else data
        return


    # validity checks
    def __bool__(self): return gsl.permutation_valid(self.data)


    # container support
    def __len__(self): return self.shape


    def __iter__(self):
        # as long as {next} succeeds
        while self.next(): #
            # the result is computed in place
            yield self
        # no more
        return


    def __getitem__(self, index):
        # reflect negative indices around the end of the permutation
        if index < 0: index = self.shape - index
        # bounds check
        if index < 0 or index >= self.shape:
            # and complain
            raise IndexError('permutation index {} out of range'.format(index))
        # get and return the element
        return gsl.permutation_get(self.data, index)


    # implementation details
    # private data
    data = None
    shape = None


# end of file
