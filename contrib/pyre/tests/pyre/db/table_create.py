#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise table creation
"""


def test():
    import pyre.db

    # declare a simple table
    class TypeFest(pyre.db.table, id="typefest"):
        """
        Exercise the type declarations
        """

        # the fields
        city = pyre.db.str()
        city.doc = "a string of arbitrary length"
        city.primary()

        state = pyre.db.str(maxlen=2)
        state.doc = "a string with maximum length 2"
        state.notNull()

        when = pyre.db.date()
        when.doc = "a date"
        when.unique()

        price = pyre.db.decimal(precision=6, scale=2)
        price.doc = "a decimal"

        pi = pyre.db.float()
        pi.doc = "a floating point number"

    # get a server
    server = pyre.db.server(name="test")
    # to build the SQL statement
    stmt = tuple(server.sql.createTable(table=TypeFest))
    # print('\n'.join(stmt))
    assert stmt == (
        "CREATE TABLE typefest",
        "  --",
        "  -- Exercise the type declarations",
        "  --",
        "(",
        "  city TEXT -- a string of arbitrary length",
        "    PRIMARY KEY,",
        "  state VARCHAR(2) DEFAULT '' -- a string with maximum length 2",
        "    NOT NULL,",
        "  when DATE -- a date",
        "    UNIQUE,",
        "  price DECIMAL(6, 2) DEFAULT 0, -- a decimal",
        "  pi DOUBLE PRECISION DEFAULT 0.0 -- a floating point number",
        ");",
        )

    return TypeFest


# main
if __name__ == "__main__":
    test()


# end of file
