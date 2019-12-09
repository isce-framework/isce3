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

    # declare the table with the measurement info
    class Measurement(pyre.db.table, id="measurement"):
        """
        Weather related measurements
        """

        # the fields
        low = pyre.db.int()
        low.doc = "the low temperature"

        high = pyre.db.int()
        high.doc = "the high temperature"

        precipitation = pyre.db.float()
        precipitation.doc = "amount of rainfall"

    # check the name
    assert Measurement.pyre_name == "measurement"
    # make sure we harvested all the descriptors (and in the right order)
    assert Measurement.pyre_localFields == tuple(
        value.field for value in (
            Measurement.low, Measurement.high, Measurement.precipitation ))
    # no inheritance here, so these should match
    assert Measurement.pyre_localFields == Measurement.pyre_fields

    # make sure all the field descriptors know Measurement as their table
    for field in Measurement.pyre_fields:
        ref = getattr(Measurement, field.name)
        assert ref.table == Measurement
        assert ref.field == field

    # now the table with the location info
    class Location(pyre.db.table, id="location"):
        """
        Location information
        """

        city = pyre.db.str()
        city.doc = "the name of the city"

        state = pyre.db.str(maxlen=2)
        state.doc = "the state"

    # check the name
    assert Location.pyre_name == "location"
    # make sure we harvested all the descriptors (and in the right order)
    assert Location.pyre_localFields == tuple(
        value.field for value in (Location.city, Location.state))
    # no inheritance here, so these should match
    assert Location.pyre_localFields == Location.pyre_fields

    # make sure all the field descriptors know Location as their table
    for field in Location.pyre_fields:
        ref = getattr(Location, field.name)
        assert ref.table == Location
        assert ref.field == field

    # now put it all together
    class Weather(Location, Measurement, id="weather"):

        date = pyre.db.date()
        date.doc = "the date of the measurement"

    # check the name
    assert Weather.pyre_name == "weather"
    # make sure we harvested all the descriptors (and in the right order)
    assert Weather.pyre_localFields == (Weather.date.field,)
    # print(Weather.pyre_fields)
    # print(tuple(field.name for field in Weather.pyre_fields))
    assert Weather.pyre_fields == tuple(
        value.field for value in (
            Weather.date,
            Weather.city, Weather.state,
            Weather.low, Weather.high, Weather.precipitation))

    # all done
    return Measurement, Location, Weather


# main
if __name__ == "__main__":
    test()


# end of file
