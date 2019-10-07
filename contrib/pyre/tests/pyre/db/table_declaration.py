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
        city.doc = "the name of the city"

        date = pyre.db.date()
        date.doc = "the date of the measurement"

        low = pyre.db.int()
        low.doc = "the low temperature"

        high = pyre.db.int()
        high.doc = "the high temperature"

        precipitation = pyre.db.float()
        precipitation.doc = "amount of rainfall"

    # check the name
    assert Weather.pyre_name == "weather"
    # make sure we harvested all the descriptors (and in the right order)
    assert Weather.pyre_localFields == tuple(
        value.field for value in (
            Weather.city, Weather.date, Weather.low, Weather.high, Weather.precipitation
            ))

    # no inheritance here, so these should match
    assert Weather.pyre_localFields == Weather.pyre_fields

    # make sure all the field descriptors report Weather as their table
    for field in Weather.pyre_fields:
        ref = getattr(Weather, field.name)
        assert ref.table == Weather
        assert ref.field == field

    # all done
    return Weather


# main
if __name__ == "__main__":
    test()


# end of file
