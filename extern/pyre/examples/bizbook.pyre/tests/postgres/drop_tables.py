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

    # access bizbook
    import bizbook

    # build a database component
    db = bizbook.pg()
    # check that we are connected to the right database
    assert db.database == 'bizbook'

    # drop the tables
    db.dropTable(bizbook.schema.InvoiceItem)
    db.dropTable(bizbook.schema.Invoice)
    db.dropTable(bizbook.schema.Editor)
    db.dropTable(bizbook.schema.Author)
    db.dropTable(bizbook.schema.Book)
    db.dropTable(bizbook.schema.Staff)
    db.dropTable(bizbook.schema.ContactMethod)
    db.dropTable(bizbook.schema.Address)
    db.dropTable(bizbook.schema.Publisher)
    db.dropTable(bizbook.schema.Person)
    db.dropTable(bizbook.schema.Location)

    # and return the component
    return db


# main
if __name__ == "__main__":
    test()


# end of file
