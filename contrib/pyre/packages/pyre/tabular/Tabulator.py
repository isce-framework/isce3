# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records


# declaration
class Tabulator(records.templater):
    """
    Metaclass that builds sheets
    """


    # types
    from .Selector import Selector as pyre_selector # override the one in {records}


    # meta-methods
    def __new__(cls, name, bases, attributes, **kwds):
        """
        Build a new worksheet
        """
        # prime the name attribute; instances are given names by the user
        attributes["pyre_name"] = name
        # build the record
        sheet = super().__new__(cls, name, bases, attributes, **kwds)
        # return the class record
        return sheet


# end of file
