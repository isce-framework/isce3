# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Element:


    # meta-methods
    def __init__(self, number, symbol, name, weight, **kwds):
        # chain up
        super().__init__(**kwds)
        # store the properties
        self.name = name
        self.symbol = symbol
        self.atomicNumber = number
        self.atomicWeight = weight
        # all done
        return


    def __str__(self):
        """
        Generate a string representation, mostly for debugging purposes
        """
        return "%s (%s) - atomic number: %d, atomic weight: %g amu" \
               % (self.name, self.symbol, self.atomicNumber, self.atomicWeight)


# end of file
