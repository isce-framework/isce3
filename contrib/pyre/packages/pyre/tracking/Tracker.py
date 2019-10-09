# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections


# class declaration
class Tracker:
    """
    Record the values a key has taken
    """


    def getHistory(self, key):
        """
        Retrieve the historical record associated with a particular {key}
        """
        return self.log[key]


    def track(self, key, node):
        """
        Add {value} to the history of {key}
        """
        self.log[key].append(node)
        return


    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # the index of historical values
        self.log = collections.defaultdict(list)
        # all done
        return


# end of file
