# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Sum:
    """
    The representation of the sum of nodes
    """

    # value management
    def getValue(self):
        """
        Compute and return my value
        """
        # easy enough
        return sum(operand.value for operand in self.operands)


# end of file
