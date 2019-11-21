# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class ForeignKey:
    """
    A field decorator the encapsulates references to table fields
    """


    # types
    from .FieldReference import FieldReference as fieldReference


    # public data
    update = None # the specified action to take when the referenced field is updated
    delete = None # the specified action to take when the referenced field is deleted
    reference = None # the table/field i refer to


    @property
    def field(self):
        """
        Return the field to which I refer
        """
        return self.reference.field


    @property
    def table(self):
        """
        Return the table to which I refer
        """
        return self.reference.table


    # meta methods
    def __init__(self, key=None, onDelete=None, onUpdate=None, **kwds):
        # chain up
        super().__init__(**kwds)

        # if {key} is already a field reference
        if isinstance(key, self.fieldReference):
            # save it
            reference = key
        # otherwise, assume {key} is a table
        else:
            # and build a default reference to it
            reference = self.fieldReference(table=key, field=None)

        # record the field reference
        self.reference = key
        # and the actions
        self.delete = onDelete
        self.update = onUpdate

        # all done
        return




# end of file
