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


def test():
    # build a database component and connect to the database specified in the local
    # configuration file
    db = pyre.db.sqlite(name="test").attach()

    # in a transaction block
    with db:
        # create the location table
        db.createTable(Location)
        # create the weather table
        db.createTable(Weather)

        # drop the weather table
        db.dropTable(Weather)
        # drop the location table
        db.dropTable(Location)

    # and return the connection and the tables
    return db, Weather, Location


# main
if __name__ == "__main__":
    test()


# end of file
