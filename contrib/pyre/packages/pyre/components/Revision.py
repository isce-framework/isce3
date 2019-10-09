# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Revision:
    """
    A helper class that stores a trait value snapshot and some meta-data
    """


    # meta-methods
    def __init__(self, value, locator, priority, **kwds):
        # chain up
        super().__init__(**kwds)
        # save
        self.value = value
        self.locator = locator
        self.priority = priority
        # all done
        return


    # private data
    __slots__ = 'value', 'locator', 'priority'


# end of file
