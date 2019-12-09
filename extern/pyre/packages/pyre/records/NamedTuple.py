# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class NamedTuple(tuple):
    """
    Storage for and access to the values of record instances
    """


    # public data; patched by my metaclass
    pyre_layout = None # the record that generates me
    pyre_extract = None # the strategy for extracting values and storing them


    # meta-methods
    def __new__(cls, record, data=None, **kwds):
        """
        Initialize a new record instance by extracting values from either {data} or {kwds}
        """
        # set up an iterable over {data} if available
        source = iter(data) if data is not None else (
            # otherwise, over {kwds}; only pull measures, not derivations
            kwds.get(item.name, item.default) for item in record.pyre_measures
            )
        # extract the values
        values = cls.pyre_extract(record=record, source=source)
        # and invoke the tuple constructor; {pyre_isConst} is set to {True} by default
        return super().__new__(cls, values)


    # private data
    __slots__ = ()


# end of file
