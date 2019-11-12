#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the package is accessible
"""


def test():
    # access the packages
    import bizbook


    # build a simple projection
    class titles(bizbook.db.query, book=bizbook.schema.Book):
        """A short query on the book table"""
        id = book.id
        title = book.title
        category = book.category
        price = book.price

    # build datastore
    db = bizbook.pg()
    # check that we are connected to the right database
    assert db.database == 'bizbook'

    # run the query
    for record in db.select(titles):
        # check the length
        assert len(record) == 4
        # check the fields
        assert hasattr(record, 'id')
        assert hasattr(record, 'title')
        assert hasattr(record, 'category')
        assert not hasattr(record, 'publisher')
        assert not hasattr(record, 'date')
        assert not hasattr(record, 'advance')
        assert hasattr(record, 'price')
        assert not hasattr(record, 'description')

    return db, titles


# main
if __name__ == "__main__":
    test()


# end of file
