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


def test():
    # import journal
    # journal.debug("postgres.init").active = True
    # journal.debug("postgres.execute").active = True
    # journal.debug("postgres.connection").active = True

    # access the bizbook package
    import bizbook

    # build a database component
    db = bizbook.pg()
    # check that we are connected to the right database
    assert db.database == 'bizbook'
    # tell postgres to shut up
    db.execute("SET client_min_messages = warning;")

    # build the tables
    db.createTable(bizbook.schema.Location)
    db.createTable(bizbook.schema.Person)
    db.createTable(bizbook.schema.Publisher)
    db.createTable(bizbook.schema.Address)
    db.createTable(bizbook.schema.ContactMethod)
    db.createTable(bizbook.schema.Staff)
    db.createTable(bizbook.schema.Book)
    db.createTable(bizbook.schema.Author)
    db.createTable(bizbook.schema.Editor)
    db.createTable(bizbook.schema.Invoice)
    db.createTable(bizbook.schema.InvoiceItem)

    # and return the component
    return db


# main
if __name__ == "__main__":
    test()


# end of file
