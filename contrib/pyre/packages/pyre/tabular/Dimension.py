# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from . import measure


# declaration
class Dimension:
    """
    The base class for implementing data binning strategies
    """


    # public data
    measure = None # the sheet descriptor to bin


    # meta-methods
    def __init__(self, measure, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my measure
        self.measure = measure
        # all done
        return


    def __get__(self, chart, cls):
        # if i am being accessed through an instance
        if chart:
            # get the journal
            import journal
            # complain
            raise journal.firewall('pyre.tabular').log(
                "dimensions can't operate on chart instances")
        # otherwise
        return self


# end of file
