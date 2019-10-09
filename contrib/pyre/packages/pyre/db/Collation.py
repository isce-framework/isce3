# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Collation:
    """
    An encapsulation of the collation sequence
    """


    # types
    from .FieldReference import FieldReference as fieldReference


    # public data
    fieldref = None
    collation = "ASC"


    # interface
    def sql(self, context=None):
        """
        Render me as an SQL expression
        """
        # NYI: i would like to be able to trim the fully qualified table reference for fields
        # that are locally aliased in the current query; see the test
        # {query_collation_explicit} for an example: the {ORDER BY} expression should just
        # mention {date}, rather {weather.date}

        # get the {mill} to render my field reference and append my collation order
        return "{} {}".format(self.fieldref.sql(context=context), self.collation)


    # meta methods
    def __init__(self, fieldref, collation=collation, **kwds):
        # chain up
        super().__init__(**kwds)

        # if necessary
        if not isinstance(fieldref, self.fieldReference):
            # convert field to references
            fieldref = self.fieldReference(field=fieldref, table=None)

        # save the field reference and the collation order
        self.fieldref = fieldref
        self.collation = collation

        # all done
        return


# end of file
