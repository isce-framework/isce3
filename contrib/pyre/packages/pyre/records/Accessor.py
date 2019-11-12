# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Selector import Selector


class Accessor(Selector):
    """
    The object responsible for providing named access to field values

    Accessors connect the name of a field to its value in the underlying tuple by keeping track
    of the index of their column. This information is available when the record class is
    formed: accessors are built and attached to the tuple generators by {Templater}, the
    {Record} metaclass
    """


    # meta-methods
    def __get__(self, record, cls):
        """
        Field retrieval
        """
        # if the target of this access is the class itself
        if record is None:
            # just return my meta-data
            return self.field

        # otherwise, retrieve my item and return it
        return record[self.index]


    def __set__(self, record, value):
        """
        Field modification
        """
        # try to set the value of this field; this will fail for const records
        record[self.index] = value
        # all done
        return


# end of file
