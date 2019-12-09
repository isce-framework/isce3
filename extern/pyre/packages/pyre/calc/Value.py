# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Value:
    """
    Mix-in class to encapsulate nodes that can hold a value.
    """


    # value management
    def getValue(self, **kwds):
        """
        Return my value
        """
        # easy enough
        return self._value


    def setValue(self, value, **kwds):
        """
        Set my value
        """
        # store the value
        self._value = value
        # all done
        return self


    # meta methods
    def __init__(self, value=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the value
        self._value = value
        # all done
        return


    # private data
    _value = None


# end of file
