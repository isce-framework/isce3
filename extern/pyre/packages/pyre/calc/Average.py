# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Average:
    """
    The representation of the average value of a collection of nodes
    """


    # value management
    def getValue(self):
        """
        Compute and return my value
        """
        # evaluate my operands
        values = tuple(operand.value for operand in self.operands)
        # compute and return the average
        return sum(values)/len(values)


# end of file
