# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class InfoZip:
    """
    Mixin that knows how to pull information from {stat} structures
    """

    # meta methods
    def __init__(self, info, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the metadata
        self.info = info
        # all done
        return


# end of file
