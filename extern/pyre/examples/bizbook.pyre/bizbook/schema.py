# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This file collects the table declarations for the {bizbook} database from

    "The Practical SQL Handbook", Second Edition
    Judith S. Bowman
    Sandra L. Emerson
    Marcy Darnovsky
"""


# access the package
import bizbook


class Location(bizbook.db.table, id="locations"):
    """
    The table of locations
    """
    id = bizbook.db.str().primary()
    address = bizbook.db.str()
    city = bizbook.db.str()
    state = bizbook.db.str()
    zip = bizbook.db.str()


class Person(bizbook.db.table, id="persons"):
    """
    The table of people
    """
    ssn = bizbook.db.str().primary()
    lastname = bizbook.db.str()
    firstname = bizbook.db.str()


class Publisher(bizbook.db.table, id="publishers"):
    """
    The book publishers
    """
    id = bizbook.db.str().primary()
    name = bizbook.db.str()
    headquarters = bizbook.db.reference(key=Location.id)


class Address(bizbook.db.table, id="addresses"):
    """
    The table of addresses
    """
    person = bizbook.db.reference(key=Person.ssn)
    address = bizbook.db.reference(key=Location.id)


class Staff(bizbook.db.table, id="staff"):
    """
    Information about employee roles
    """
    person = bizbook.db.reference(key=Person.ssn)
    position = bizbook.db.str()


class ContactMethod(bizbook.db.table, id="contact_methods"):
    """
    Contact information
    """
    uid = bizbook.db.str()
    method = bizbook.db.str()
    person = bizbook.db.reference(key=Person.ssn)


class Book(bizbook.db.table, id="books"):
    """
    Books
    """
    id = bizbook.db.str().primary()
    title = bizbook.db.str()
    category = bizbook.db.str()
    publisher = bizbook.db.reference(key=Publisher.id)
    date = bizbook.db.str()
    price = bizbook.db.decimal(precision=11, scale=2)
    advance = bizbook.db.decimal(precision=8, scale=2)
    description = bizbook.db.str()


class Author(bizbook.db.table, id="authors"):
    """
    Author information
    """
    author = bizbook.db.reference(key=Person.ssn)
    book = bizbook.db.reference(key=Book.id)
    ordinal = bizbook.db.int()
    share = bizbook.db.decimal(precision=4, scale=3)


class Editor(bizbook.db.table, id="editors"):
    """
    Editor information
    """
    editor = bizbook.db.reference(key=Person.ssn)
    book = bizbook.db.reference(key=Book.id)
    ordinal = bizbook.db.int()


class Invoice(bizbook.db.table, id="invoices"):
    """
    Invoices
    """
    id = bizbook.db.str().primary()
    client = bizbook.db.str()
    po = bizbook.db.str()
    date = bizbook.db.str()


class InvoiceItem(bizbook.db.table, id="invoice_item"):
    """
    Invoice line items
    """
    invoice = bizbook.db.reference(key=Invoice.id)
    book = bizbook.db.reference(key=Book.id)
    ordered = bizbook.db.int()
    shipped = bizbook.db.int()
    date = bizbook.db.str()


# end of file
