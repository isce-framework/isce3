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

    # declare a simple table
    class Weather(pyre.db.table, id="weather"):
        """
        The sample table from the postgres tutorial
        """
        # the fields
        city = pyre.db.str()
        date = pyre.db.date()
        low = pyre.db.int()
        high = pyre.db.int()
        precipitation = pyre.db.float()

    # and a simple query
    class measurements(pyre.db.query, weather=Weather):
        # the fields
        city = weather.city
        date = weather.date

    # get a server
    server = pyre.db.server()
    # generate the SQL statement
    stmt = tuple(server.sql.select(measurements))
    # print('\n'.join(stmt))
    assert stmt == (
        "SELECT",
        "    weather.city AS city,",
        "    weather.date AS date",
        "  FROM",
        "    weather;"
        )

    # all done
    return Weather


# main
if __name__ == "__main__":
    test()


# end of file
