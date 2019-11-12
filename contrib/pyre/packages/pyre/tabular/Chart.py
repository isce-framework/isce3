# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# metaclass
from .Surveyor import Surveyor


# declaration
class Chart(metaclass=Surveyor):
    """
    The base class for imposing coördinate systems on sheets

    A chart contains the specification of a number of dimensions that enable the categorization
    and analysis of the facts in a sheet. For example, given a sales table that contains
    transaction information that includes date, sku and amount, a chart with these three
    dimensions would simplify answering questions such as "compute the total sales of a given
    sku in a given time period".

    Charts are used by pivot tables as a means of imposing structure on the data and
    precomputing data slices. See {pyre.tabular.Pivot} and the {pyre.tabular.Dimension}
    subclasses for more details.
    """


    # public data
    # class attributes, common to all instances of a given chart
    pyre_sheets = None # map of local aliases to sheets
    pyre_dimensions = None # the complete list of my dimensions
    pyre_localDimensions = None # the locally declared ones


    # interface
    def pyre_filter(self, **kwds):
        """
        Create an iterable over those facts that statisfy the criteria specified in {kwds},
        which is assumed to be a value specification for each dimension that is to be used to
        restrict the data set
        """
        # identify the relevant bins
        bins = (getattr(self, name)[value] for name, value in kwds.items())
        # build and return the restriction
        return set.intersection(*bins)


    # meta-methods
    def __init__(self, sheet, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the sheet i am bound to
        self.sheet = sheet
        # all done
        return


# end of file
