#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise updating rows in tables
"""


def test():
    # access the package
    import pyre.db

    # declare the person table
    class Person(pyre.db.table, id='persons'):

        id = pyre.db.int().primary()
        name = pyre.db.str().notNull()
        phone = pyre.db.str().notNull()
        birthday = pyre.db.str()

    # initialize a person template
    eva = pyre.db.template(Person)
    # record some assignments
    eva.phone = '1 800 555 7687'
    eva.birthday = '1930/10/28'

    # get a server
    server = pyre.db.server(name="test")
    # generate the SQL statement that updates the customer table
    stmt = tuple(server.sql.updateRecords(template=eva, condition=(Person.name == 'Eva Lu Ator')))
    # print('\n'.join(stmt))
    assert stmt == (
        "UPDATE persons",
        "  SET",
        "    (phone, birthday) = ('1 800 555 7687', '1930/10/28')",
        "  WHERE ((name) = ('Eva Lu Ator'));"
        )

    return


# main
if __name__ == "__main__":
    test()


# end of file
