# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Printer:
    """
    A stream filter that prints the objects that pass through it
    """


    def __call__(self, stream):
        """
        Print the objects in {stream} and pass them through
        """
        # loop through the iterable
        for item in stream:
            # print the item
            print(item)
            # pass the item through
            yield item
        # all done
        return


# end of file
