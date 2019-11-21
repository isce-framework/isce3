# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Literal:
    """
    Class that encapsulates values encountered in expressions that are not instances of members
    of the {Node} class hierarchy.
    """


    # constants
    category = 'literal'


    # interface
    @property
    def literals(self):
        """
        Return a sequence of the literals in my span
        """
        # i am one
        yield self
        # and nothing further
        return


    # meta-methods
    def __init__(self, value, **kwds):
        # chain up
        super().__init__(**kwds)
        # store the foreign object as my value
        self._value = value
        # all done
        return


    # support for graph traversals
    def identify(self, authority, **kwds):
        """
        Let {authority} know I am a literal
        """
        # invoke the callback
        return authority.onLiteral(literal=self, **kwds)


# end of file
