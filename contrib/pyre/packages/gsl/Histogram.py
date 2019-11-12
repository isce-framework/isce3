# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from . import gsl


# the class declaration
class Histogram:
    """
    A wrapper over a gsl histogram
    """


    # types
    from .Vector import Vector as vector


    # public data
    bins = 0


    # interface
    # specifying the binning strategy
    def uniform(self, lower, upper):
        """
        Adjust the histogram bins to cover the range [lower, upper) uniformly. The values of
        the bins are reset to zero
        """
        # adjust the bins
        gsl.histogram_uniform(self.data, lower, upper)
        # and return
        return self


    def ranges(self, points):
        """
        Use {points} to define the histogram bins. {points} is expected to be an iterable of
        size one greater than the size of the histogram itself; the extra element is used to
        specify the upper value of the last bin. The entries in {points} must be monotonically
        increasing
        """
        # adjust the bins
        gsl.histogram_ranges(self.data, tuple(points))
        # and return
        return self


    # adjusting the bin values
    def reset(self):
        """
        Reset the values of all the bins to zero
        """
        # reset
        gsl.histogram_reset(self.data)
        # and return
        return self


    def increment(self, x):
        """
        Increment by one the bin whose range contains {x}
        """
        # do it
        gsl.histogram_increment(self.data, x)
        # and return
        return self


    def accumulate(self, x, weight):
        """
        Add {weight} to the bin whose range contains {x}
        """
        # do it
        gsl.histogram_accumulate(self.data, x, weight)
        # and return
        return self


    def fill(self, values):
        """
        Increment my frequency counts using the contents of the vector {values}
        """
        # we have a function for that
        gsl.histogram_fill(self.data, values.data)
        # and return
        return self


    # copying
    def clone(self):

        """
        Allocate a new histogram and initialize it using my values
        """
        # build the clone
        clone = type(self)(bins=self.bins, data=gsl.histogram_clone(self.data))
        # and return it
        return clone


    def copy(self, other):
        """
        Make me an exact copy of {other}
        """
        # have the extension initialize me as a copy
        gsl.histogram_copy(self.data, other.data)
        # and return it
        return clone


    def values(self):
        """
        Return a vector that contains the values from each of my bins. This is equivalent to

          v = gsl.vector.alloc(shape=self.bins)
          for i in range(self.bins):
              v[i] = self[i]
          return v
        """
        # fill the vector data
        data = gsl.histogram_vector(self.data)
        # allocate the vector and return it
        return self.vector(shape=self.bins, data=data)


    # operations
    def find(self, x):
        """
        Return the index of the bin that contains the value {x}
        """
        # easy enough
        return gsl.histogram_find(self.data, x)


    def max(self):
        """
        Return my maximum upper range
        """
        # easy enough
        return gsl.histogram_max(self.data)


    def min(self):
        """
        Return my minimum lower range
        """
        # easy enough
        return gsl.histogram_min(self.data)


    def range(self, i):
        """
        Return a tuple [lower, upper) that describes the range of the {i}th bin
        """
        # easy enough
        return gsl.histogram_range(self.data, i)


    # statistics
    def max_bin(self):
        """
        Return the index of the bin where maximum value is contained in the histogram
        """
        # easy enough
        return gsl.histgram_max_bin(self.data)


    def max_value(self):
        """
        Return the maximum value contained in the histogram
        """
        # easy enough
        return gsl.histgram_max_value(self.data)


    def min_bin(self):
        """
        Return the index of the bin where minimum value is contained in the histogram
        """
        # easy enough
        return gsl.histgram_min_bin(self.data)


    def min_value(self):
        """
        Return the minimum value contained in the histogram
        """
        # easy enough
        return gsl.histgram_min_value(self.data)


    def mean(self):
        """
        Return the mean of the histogrammed variable
        """
        # easy enough
        return gsl.histgram_mean(self.data)


    def sdev(self):
        """
        Return the standard deviation of the histogrammed variable
        """
        # easy enough
        return gsl.histgram_sdev(self.data)


    def sum(self):
        """
        Return the sum of all bin values
        """
        # easy enough
        return gsl.histgram_sum(self.data)


    # meta methods
    def __init__(self, bins, data=None, **kwds):
        super().__init__(**kwds)
        self.bins = bins
        self.data = gsl.histogram_alloc(bins) if data is None else data
        return


    # container support
    def __len__(self): return self.bins


    def __iter__(self):
        # for each valid value of the index
        for index in range(self.bins):
            # produce the corresponding count
            yield gsl.histogram_get(self.data, index)
        # all done
        return


    def __getitem__(self, index):
        # get and return the element
        return gsl.histogram_get(self.data, index)


    # in-place arithmetic
    def __iadd__(self, other):
        """
        In-place addition with the elements of {other}
        """
        # if other is a histogram
        if type(other) is type(self):
            # do histogram-histogram addition
            gsl.histogram_add(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do constant addition
            gsl.histogram_shift(self.data, float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    def __isub__(self, other):
        """
        In-place subtraction with the elements of {other}
        """
        # if other is a histogram
        if type(other) is type(self):
            # do histogram-histogram subtraction
            gsl.histogram_sub(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do constant subtraction
            gsl.histogram_shift(self.data, -float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    def __imul__(self, other):
        """
        In-place multiplication with the elements of {other}
        """
        # if other is a histogram
        if type(other) is type(self):
            # do histogram-histogram multiplication
            gsl.histogram_mul(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do scaling by constant
            gsl.histogram_scale(self.data, float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    def __itruediv__(self, other):
        """
        In-place addition with the elements of {other}
        """
        # if other is a histogram
        if type(other) is type(self):
            # do histogram-histogram division
            gsl.histogram_div(self.data, other.data)
            # and return
            return self
        # if other is a number
        if isinstance(other, numbers.Number):
            # do scaling by constant
            gsl.histogram_scale(self.data, 1/float(other))
            # and return
            return self
        # otherwise, let the interpreter know
        raise NotImplemented


    # implementation details
    # private data
    data = None


# end of file
