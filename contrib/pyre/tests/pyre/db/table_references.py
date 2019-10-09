#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise table declaration
"""


def test():
    # access to the package
    import pyre.db

    # the foreign table
    class City(pyre.db.table, id="cities"):
        """
        A table of cities
        """
        # the fields
        id = pyre.db.int().primary()
        name = pyre.db.str()
        state = pyre.db.str(maxlen=2)


    # declare a simple table
    class Weather(pyre.db.table, id="weather"):
        """
        The sample table from the postgres tutorial
        """

        # the fields
        city = pyre.db.reference(key=City.id)

        date = pyre.db.date()
        low = pyre.db.int()
        high = pyre.db.int()
        precipitation = pyre.db.float()

    # get a server
    server = pyre.db.server(name="test")
    # to build the SQL statement
    stmt = tuple(server.sql.createTable(table=Weather))
    # print('\n'.join(stmt))
    assert stmt == (
        "CREATE TABLE weather",
        "  --",
        "  -- The sample table from the postgres tutorial",
        "  --",
        "(",
        "  city INTEGER",
        "    REFERENCES cities (id),",
        "  date DATE,",
        "  low INTEGER DEFAULT 0,",
        "  high INTEGER DEFAULT 0,",
        "  precipitation DOUBLE PRECISION DEFAULT 0.0",
        ");"
        )
    # all done
    return Weather, City


# main
if __name__ == "__main__":
    test()


# end of file
