# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Sequence:
    """
    Mix-in class that forms the basis of the representation of sequences
    """


    # constants
    category = "sequence"


    # classifiers
    @property
    def sequences(self):
        """
        Return a sequence over sequences in my dependency graph
        """
        # i am one
        yield self
        # nothing further
        return


    # value management
    def getValue(self, **kwds):
        """
        Compute and return my value
        """
        # return a tuple with the value of each operand; do not be tempted to avoid realizing
        # the container: value memoization will store the generator, it will get exhausted on
        # first read, and the value of the sequence will be empty thereafter
        return tuple(op.value for op in self.operands)


# end of file
