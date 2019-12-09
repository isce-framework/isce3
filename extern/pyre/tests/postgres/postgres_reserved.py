#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that the table attributes we use to specify query parameters are treated reserved
keywords by the back end, which makes it unnecessary to place them in the {pyre_} namespace.
"""


import pyre.db


def test():

    # connect to the test database
    db = pyre.db.postgres(name="test").attach()

    # order
    class invalid(pyre.db.table):
        order = pyre.db.str()

    try:
        db.createTable(invalid)
        assert False
    except db.exceptions.ProgrammingError:
        pass

    # where
    class invalid(pyre.db.table):
        where = pyre.db.str()

    try:
        db.createTable(invalid)
        assert False
    except db.exceptions.ProgrammingError:
        pass

    # group
    class invalid(pyre.db.table):
        group = pyre.db.str()

    try:
        db.createTable(invalid)
        assert False
    except db.exceptions.ProgrammingError:
        pass

    # and return the connection
    return db


# main
if __name__ == "__main__":
    test()


# end of file
