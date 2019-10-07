# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Reference:
    """
    A node that refers to another node
    """

    # constants
    category = 'reference'


    # classifiers
    @property
    def references(self):
        """
        Return a sequence over the nodes in my dependency graph that are references to other nodes
        """
        # i am one
        yield self
        # nothing further
        return


    # value management
    def getValue(self):
        """
        Compute and return my value
        """
        # get my referent
        referent, = self.operands
        # and ask him for his value
        return referent.value


    def setValue(self, value):
        """
        Set the value of the node i refer to
        """
        # get my referent
        referent, = self.operands
        # set its value
        referent.value = value
        # all done
        return self



    # support for graph traversals
    def identify(self, authority, **kwds):
        """
        Let {authority} know I am a reference
        """
        # invoke the callback
        return authority.onReference(reference=self, **kwds)


# end of file
