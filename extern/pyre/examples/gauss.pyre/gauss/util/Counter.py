# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Counter:
    """
    A stream filter that counts the number of objects that pass through it
    """


    # public data
    count = 0


    # meta methods
    def __init__(self, start=0, **kwds):
        """
        Build a counter that start counting at {start}
        """
        super().__init__(**kwds)
        self.count = start
        return


    def __call__(self, stream):
        """
        Count the number of objects in {stream} and pass them through
        """
        # loop through the iterable
        for item in stream:
            # update the count
            self.count += 1
            # pass the item through
            yield item
        # all done
        return


# end of file
