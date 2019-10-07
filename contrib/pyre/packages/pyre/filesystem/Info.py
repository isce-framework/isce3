# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Info:
    """
    Base class for encapsulating nodal metadata for filesystem entries
    """


    # meta methods
    def __init__(self, uri, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the node uri
        self.uri = uri
        # all done
        return


# end of file
