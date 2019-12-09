# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records


# declaration
class FieldSelector(records.selector):
    """
    Access to the field descriptors
    """


    # types
    from .FieldReference import FieldReference as fieldReference


    # meta-methods
    def __get__(self, record, cls):
        """
        Field retrieval
        """
        # return a field reference regardless of the access target
        return self.fieldReference(table=cls, field=self.field)


# end of file
