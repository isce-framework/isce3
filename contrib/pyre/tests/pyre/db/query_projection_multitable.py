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

    # and a meaningless query from two copies of the same table
    class measurements(pyre.db.query, w1=Weather, w2=Weather):
        # the fields
        city = w1.city
        date = w2.date

    # get a server
    server = pyre.db.server()
    # generate the SQL statement
    stmt = tuple(server.sql.select(measurements))
    # print('\n'.join(stmt))
    assert stmt == (
        "SELECT",
        "    w1.city AS city,",
        "    w2.date AS date",
        "  FROM",
        "    weather AS w1,",
        "    weather AS w2;"
        )

    # all done
    return Weather


# main
if __name__ == "__main__":
    test()


# end of file
