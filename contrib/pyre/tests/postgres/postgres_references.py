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

class Location(pyre.db.table, id="location"):

    id = pyre.db.int().primary()
    id.doc = "the unique key that identifies a location"

    latitude = pyre.db.float()
    latitude.doc = "the latitude of the location"

    longitude = pyre.db.float()
    longitude.doc = "the longitude of the location"


class Weather(pyre.db.table, id="weather"):

    location = pyre.db.reference(
        key=Location.id, onDelete=pyre.db.setNull, onUpdate=pyre.db.setDefault)
    location.doc = "the location of the measurements"

    date = pyre.db.date()
    date.doc = "the date of the measurement"

    low = pyre.db.decimal(precision=5, scale=2)
    low.doc = "the temperature low"

    high = pyre.db.decimal(precision=5, scale=2)
    high.doc = "the temperature low"


def queryForTable(db, table):
    """
    Interrogate the table of tables looking for {table}
    """
    # build the statement
    sql = "SELECT tablename FROM pg_tables WHERE tablename='{}'".format(table.pyre_name)
    # execute it and return the result
    return db.execute(sql)


def test():
    import journal
    # journal.debug("postgres.init").active = True
    # journal.debug("postgres.execute").active = True
    # journal.debug("postgres.connection").active = True
    journal.debug("postgres.transactions").active = True

    # build a database component and connect to the database specified in the local
    # configuration file
    db = pyre.db.postgres(name="test").attach()

    # ask {libpq}  to be quiet
    db.execute("SET client_min_messages = warning;")
    # in a transaction block
    with db:
        # create the location table
        db.createTable(Location)
        # verify it is there
        assert queryForTable(db, Location) == (('tablename',), (Location.pyre_name,))

        # create the weather table
        # print('\n'.join(db.sql.createTable(Weather)))
        db.createTable(Weather)
        # verify it is there
        assert queryForTable(db, Weather) == (('tablename',), (Weather.pyre_name,))

        # drop the weather table
        db.dropTable(Weather)
        # and check it's gone
        assert queryForTable(db, Weather) == (('tablename',),)

        # drop the location table
        # print('\n'.join(db.sql.createTable(Location)))
        db.dropTable(Location)
        # and check it's gone too
        assert queryForTable(db, Location) == (('tablename',),)

    # and return the connection and the tables
    return db, Weather, Location


# main
if __name__ == "__main__":
    test()


# end of file
