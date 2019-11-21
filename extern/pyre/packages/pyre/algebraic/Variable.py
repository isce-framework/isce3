# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Variable:
    """
    Mix-in class to encapsulate nodes
    """


    # constants
    category = 'variable'


    # interface
    @property
    def span(self):
        """
        Return a sequence of the nodes in my span
        """
        # i am one
        yield self
        # and nothing further
        return


    @property
    def variables(self):
        """
        Return a sequence of the variables in my span
        """
        # i am one
        yield self
        # and nothing further
        return


    # support for graph traversals
    def identify(self, authority, **kwds):
        """
        Let {authority} know I am a variable
        """
        # invoke the callback
        return authority.onVariable(variable=self, **kwds)


# end of file
