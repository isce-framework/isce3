# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Unresolved:
    """
    A node that raises {UnresolvedNodeError} when its value is read
    """


    # exceptions
    from .exceptions import UnresolvedNodeError


    # constants
    category = 'unresolved'
    # public data
    request = None # the unresolved name


    # classifiers
    @property
    def unresolveds(self):
        """
        Return a sequence over the unresolved nodes in my dependency graph
        """
        # i am one
        yield self
        # nothing further
        return


    # value management
    def getValue(self):
        """
        Compute my value
        """
        # asking for my value is an error
        raise self.UnresolvedNodeError(node=self, name=self.request)


    # support for graph traversals
    def identify(self, authority, **kwds):
        """
        Let {authority} know I am an unresolved node
        """
        # invoke the callback
        return authority.onUnresolved(unresolved=self, **kwds)


    # meta methods
    def __init__(self, request, **kwds):
        # chain up
        super().__init__(**kwds)
        # store the name of the requested node
        self.request = request
        # all done
        return


    def __str__(self):
        # i have a name...
        return self.request


    # debugging support
    def dump(self, name, indent):
        print('{}{}: <unresolved>'.format(indent, name))
        return self


# end of file
