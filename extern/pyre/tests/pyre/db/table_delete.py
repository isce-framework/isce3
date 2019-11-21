#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise inserting rows in tables
"""


def test():
    # access the package
    import pyre.db

    # declare the person table
    class Person(pyre.db.table, id='persons'):

        id = pyre.db.int().primary()
        name = pyre.db.str().notNull()
        phone = pyre.db.str(maxlen=10).notNull()


    # get a server
    server = pyre.db.server(name="test")

    # attempt to drop a {Person} record
    stmt = tuple(server.sql.deleteRecords(table=Person, condition=(Person.name=='Bit Twiddle')))
    assert stmt == (
        "DELETE FROM persons",
        "  WHERE ((name) = ('Bit Twiddle'));"
        )

    # attempt to remove all of them
    stmt = tuple(server.sql.deleteRecords(table=Person, condition=None))
    # print('\n'.join(stmt))
    assert stmt == (
        "DELETE FROM persons;",
        )

    return


# main
if __name__ == "__main__":
    test()


# end of file
