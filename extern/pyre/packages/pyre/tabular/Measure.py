# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records
from .. import schemata


# declaration
@schemata.typed
class Measure(records.measure):
    """
    Base class for the measures in this package
    """


    # interface
    def primary(self):
        """
        Mark this measure as a primary key
        """
        # mark me
        self._primary = True
        # and return me so I chain properly
        return self


    # private data
    _primary = False # {True} when this measure is a primary key used to create an index


# end of file
