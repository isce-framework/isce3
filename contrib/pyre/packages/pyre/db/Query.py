# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records
# metaclass
from .Selector import Selector


# declaration
class Query(records.record, hidden=True, metaclass=Selector):
    """
    Base class for describing database queries
    """

    # public data
    where = None # retrieve only rows that satisfy this expression
    order = None # control over the sorting order of the results
    group = None # aggregate the results using the distinct values of this column

    # metaclass decorations; treat as read-only
    pyre_tables = {} # a map of local names to referenced tables


# end of file
