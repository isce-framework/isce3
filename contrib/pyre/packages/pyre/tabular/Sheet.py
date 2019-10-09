# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records
# metaclass
from .Tabulator import Tabulator


# declaration
class Sheet(records.record, metaclass=Tabulator):
    """
    The base class for pyre worksheets, collections of record instances
    """


    # public data
    pyre_name = None
    pyre_data = None # the list of records


    # interface
    def pyre_immutable(self, data):
        """
        Iterate over {data} extracting records that are compatible with my layout and use them to
        populate my data set
        """
        # iterate over the records in {data}
        for entry in data:
            # convert the {row} into a mutable tuple
            row = super().pyre_immutable(data=entry)
            # populate the data set
            self.pyre_append(row=row)
        # all done
        return self


    def pyre_mutable(self, data):
        """
        Iterate over {data} extracting records that are compatible with my layout and use them to
        populate my data set
        """
        # iterate over the records in {data}
        for entry in data:
            # convert the {row} into a mutable tuple
            row = super().pyre_mutable(data=entry)
            # populate the data set
            self.pyre_append(row=row)
        # all done
        return self


    def pyre_append(self, row):
        """
        Add the given {row} to my data set
        """
        # get my dataset
        dataset = self.pyre_data
        # add the record to the dataset
        dataset.append(row)
        # all done
        return self


    def pyre_new(self):
        """
        Create a new blank mutable record instance and add it to my data set
        """
        # ask my record to build one
        record = super().pyre_mutable()
        # add it to the pile
        self.pyre_append(record)
        # and hand it to the caller
        return record


    @classmethod
    def pyre_offset(cls, measure):
        """
        Return the column number of {measure}
        """
        # easy enough
        getattr(cls, measure).index


    # meta-methods
    def __init__(self, name, **kwds):
        # chain up
        super().__init__(**kwds)
        # set my name
        self.pyre_name = name
        # initialize my data set
        self.pyre_data = []
        # all done
        return


    def __len__(self):
        """
        Compute the number of records in my dataset
        """
        # delegate to the dataset
        return len(self.pyre_data)


    def __iter__(self):
        """
        Build an iterator over my data set
        """
        # delegate to the dataset
        return iter(self.pyre_data)


    def __getitem__(self, address):
        """
        Retrieve the portion of the sheet that corresponds to {address}
        """
        # by definition, string addresses refer to my columns
        if isinstance(address, str):
            # let my descriptor do the work
            return getattr(self, address)

        # again by definition, single integers refer to my rows
        if isinstance(address, int):
            # retrieve the requested row and return it
            return self.pyre_data[address]

        # eventually I will support really smart slicing
        raise NotImplementedError('NYI: smart sheet indexing is not yet implemented')


# end of file
