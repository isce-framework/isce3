# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Operator:
    """
    Mix-in class that forms the basis of the representation of operations among nodes
    """


    # constants
    category = 'operator'

    # public data
    evaluator = None


    # meta-methods
    def __init__(self, evaluator, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my operator
        self.evaluator = evaluator
        # all done
        return


    # support for graph traversals
    def identify(self, authority, **kwds):
        """
        Let {authority} know I am an operator
        """
        # invoke the callback
        return authority.onOperator(operator=self, **kwds)


# end of file
