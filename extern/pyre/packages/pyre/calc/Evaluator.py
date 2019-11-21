# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Evaluator:
    """
    Mix-in class that computes the value of operator nodes by invoking their evaluator on their
    operands
    """


    # value management
    def getValue(self, **kwds):
        """
        Compute and return my value
        """
        # compute the values of my operands
        values = (op.value for op in self.operands)
        # apply my operator
        return self.evaluator(*values)


# end of file
