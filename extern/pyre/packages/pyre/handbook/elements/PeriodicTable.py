# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class PeriodicTable:
    """
    An encapsulation of the periodic table of elements
    """


    # interface
    def name(self, name):
        """
        Retrieve the element with the given {name}
        """
        # look it up
        return self._nameIndex[name.lower()]


    def symbol(self, symbol):
        """
        Retrieve the element with the given {symbol}
        """
        # look it up
        return self._symbolIndex[symbol.lower()]


    def atomicNumber(self, z):
        """
        Retrieve the element with the given atomic number
        """
        # look it up
        return self._atomicNumberIndex[z-1]


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # load the element database
        from .elements import elements
        # build the indices
        self._atomicNumberIndex = elements
        self._nameIndex = { element.name : element for element in elements }
        self._symbolIndex = { element.symbol : element for element in elements }
        # all done
        return


# end of file
