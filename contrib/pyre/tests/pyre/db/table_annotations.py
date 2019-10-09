#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise attaching constraints to tables
"""


def test():
    # access the package
    import pyre.db

    # declare the customer table
    class Customer(pyre.db.table, id='customers'):
        """
        Simple customer table
        """
        cid = pyre.db.int(default=None)
        cid.doc = "the customer id"
        name = pyre.db.str().notNull()
        phone = pyre.db.str(maxlen=10)
        balance = pyre.db.decimal(precision=7, scale=2)

    # mark {cid} as a  primary key
    Customer.pyre_primaryKey(Customer.cid)
    # and {phone} as unique
    Customer.pyre_unique(Customer.name)
    # finally, attach some nameless constraints
    Customer.pyre_check(Customer.balance >= 0)
    Customer.pyre_check(Customer.balance < 10000)

    # declare the order table
    class Order(pyre.db.table, id='orders'):
        """
        Trivial order table
        """
        oid = pyre.db.int().primary()
        cid = pyre.db.int(default=None)

    # mark {cid} as a foreign key
    Order.pyre_foreignKey(field=Order.cid, foreign=Customer.cid)

    # get a server
    server = pyre.db.server(name="test")

    # generate the SQL statement that creates the customer table
    stmt = tuple(server.sql.createTable(table=Customer))
    #print('\n'.join(stmt))
    assert stmt == (
        "CREATE TABLE customers",
        "  --",
        "  -- Simple customer table",
        "  --",
        "(",
        "  cid INTEGER, -- the customer id",
        "  name TEXT DEFAULT ''",
        "    NOT NULL,",
        "  phone VARCHAR(10) DEFAULT '',",
        "  balance DECIMAL(7, 2) DEFAULT 0,",
        "",
        "  PRIMARY KEY (cid),",
        "  UNIQUE (name),",
        "  CHECK ((balance) >= (0)),",
        "  CHECK ((balance) < (10000))",
        ");"
        )

    # and the SQL statement that creates the order table
    stmt = tuple(server.sql.createTable(table=Order))
    # print('\n'.join(stmt))
    assert stmt == (
        "CREATE TABLE orders",
        "  --",
        "  -- Trivial order table",
        "  --",
        "(",
        "  oid INTEGER",
        "    PRIMARY KEY,",
        "  cid INTEGER,",
        "",
        "  FOREIGN KEY (cid) REFERENCES customers (cid)",
        ");"
        )

    return


# main
if __name__ == "__main__":
    test()


# end of file
