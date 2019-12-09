#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate the postgres component
"""


import pyre.db

class Weather(pyre.db.table, id="weather"):

    city = pyre.db.str()
    city.doc = "the city name"

    state = pyre.db.str(maxlen=2)
    state.doc = "the state name"

    date = pyre.db.date()
    date.doc = "the date of the measurement"

    low = pyre.db.decimal(precision=5, scale=2)
    low.doc = "the temperature low"

    high = pyre.db.decimal(precision=5, scale=2)
    high.doc = "the temperature low"


def test():
    import journal
    # journal.debug("postgres.init").active = True
    # journal.debug("postgres.execute").active = True
    # journal.debug("postgres.connection").active = True
    journal.debug("postgres.transactions").active = True

    # this is the SQL statement that looks for a table by a given name
    sql = "SELECT tablename FROM pg_tables WHERE tablename='{}'".format(Weather.pyre_name)

    # build a database component and connect to the database specified in the local
    # configuration file
    db = pyre.db.postgres(name="test").attach()

    # in a transaction block
    with db:
        # create the table
        db.createTable(Weather)
        # verify it is there
        assert db.execute(sql) == (('tablename',), ('weather',))
        # drop the table
        db.dropTable(Weather)
        # verify it is not there
        assert db.execute(sql) == (('tablename',),)

    # and return the connection and the table
    return db, Weather


# main
if __name__ == "__main__":
    test()


# end of file
