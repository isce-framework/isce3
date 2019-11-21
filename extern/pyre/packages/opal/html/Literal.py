# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Literal:
    """
    Representation of all non-HTML content, such as literal text
    """


    # public data
    value = None # the encapsulated content


    # document traversal
    def identify(self, inspector, **kwds):
        """
        The second half of double dispatch to the inspector's handler for this object
        """
        # dispatch
        return inspector.onLiteral(element=self, **kwds)


    # meta-methods
    def __init__(self):
        # storage for my contents
        self.value = []
        # all done
        return


# end of file
