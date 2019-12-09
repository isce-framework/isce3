# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from ..patterns.AttributeClassifier import AttributeClassifier


# declaration
class Surveyor(AttributeClassifier):
    """
    Inspect charts and harvest their dimensions
    """


    # types
    from .Dimension import Dimension as pyre_dimension
    from .Tabulator import Tabulator as pyre_tabulator


    # meta-methods
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        """
        Prepare a container for the attributes of a chart by looking through {kwds} for sheet
        aliases and making them available as class attributes during the chart
        declaration. This makes it possible to refer to sheets when setting up the chart
        dimensions
        """
        # make a pile of sheets
        sheets = dict(
            # mapping names to sheets
            (name, sheet) for name, sheet in kwds.items()
            # for every entry that is a sheet
            if isinstance(sheet, cls.pyre_tabulator))

        # my machinery is not smart enough to handle charts over multiple sheets. yet.
        if len(sheets) > 1:
            # get the journal
            import journal
            # complain
            raise journal.firewall('pyre.tabular').log('charts need precisely one sheet')

        # remove all the sheets from {kwds}
        for name in sheets: del kwds[name]

        # make the attribute container
        attributes = super().__prepare__(name, bases, **kwds)
        # add the sheets as individual variables
        attributes.update(sheets)
        # and as a name index, if present
        if sheets: attributes['pyre_sheets'] = sheets

        # return the attributes
        return attributes


    def __new__(cls, name, bases, attributes, **kwds):
        """
        Build a new chart class record
        """
        # build the record
        chart = super().__new__(cls, name, bases, attributes, **kwds)

        # make a pile for the locally declared dimensions
        local = []
        # harvest the local dimensions
        for alias, dimension in cls.pyre_harvest(attributes, cls.pyre_dimension):
            # and add them to the pile
            local.append(dimension)
        # attach them
        chart.pyre_localDimensions = tuple(local)

        # now scan ancestors and accumulate the entire set of dimensions
        dimensions = []
        # for each base class
        for base in reversed(chart.__mro__):
            # skip the bases that are not charts
            if not isinstance(base, cls): continue
            # add the dimensions declared locally in this base to the pile
            dimensions.extend(base.pyre_localDimensions)
        # attach them
        chart.pyre_dimensions = tuple(dimensions)

        # all done; return the chart
        return chart


# end of file
