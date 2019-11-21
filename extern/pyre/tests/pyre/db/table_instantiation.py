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
    # externals
    import time
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

        low = pyre.db.decimal(precision=5, scale=2)
        low.doc = "the low temperature"

        high = pyre.db.decimal(precision=5, scale=2)
        high.doc = "the high temperature"

        precipitation = pyre.db.decimal(precision=3, scale=1)
        precipitation.doc = "amount of rainfall"

    # build one
    measurement = Weather.pyre_immutable(
        city='Los Angeles', date='2011-07-04', low=62, high=71, precipitation=0
        )

    # verify it
    assert measurement.city == 'Los Angeles'
    assert str(measurement.date) == '2011-07-04'
    assert measurement.low == 62
    assert measurement.high == 71
    assert measurement.precipitation == 0

    # all done
    return measurement


# main
if __name__ == "__main__":
    test()


# end of file
