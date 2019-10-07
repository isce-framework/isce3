# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Element:
    """
    The base class for all HTML elements
    """


    # public data
    tag = None # the element tag, i.e. "div", "p", "table"
    attributes = None # a dictionary that maps element attributes to their values


    # document traversal
    def identify(self, inspector, **kwds):
        """
        The second half of double dispatch to the inspector's handler for this object
        """
        # enforce the subclass obligation
        raise NotImplementedError("class {.__name__} must implement 'identify'".format(type(self)))


    # meta-methods
    def __init__(self, tag, **kwds):
        # record the tag
        self.tag = tag
        # and the attributes
        self.attributes = dict(kwds)
        # all done
        return


# end of file
