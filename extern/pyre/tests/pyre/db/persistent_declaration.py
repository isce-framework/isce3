#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the declaration of a persistent object
"""


def test():
    # access to the package
    import pyre.db

    # the schema
    class entity(pyre.db.table):
        """Entities"""
        eid = pyre.db.str().primary()
        name = pyre.db.str().notNull()

    class person(entity, id='persons'):
        """People"""

    class company(entity, id='companies'):
        """Companies"""

    class employement(pyre.db.table, id="employment"):
        """Relationships between people and companies"""
        eid = pyre.db.str().primary()
        employee = pyre.db.reference(key=person.eid).notNull()
        employer = pyre.db.reference(key=company.eid).notNull()
        rate = pyre.db.float()

    # the model
    class Employee(pyre.db.object):
        """
        An object whose attributes are stored in a relational schema
        """
        # my unique identifier
        eid = person.eid
        # my attributes
        name = person.name
        employer = None # how do I refer to {Employer} defined below?

    class Employer(pyre.db.object):
        """
        A more complicated object whose attributes are scattered in a schema
        """
        # my unique identifier
        eir = company.eid
        # my attributes
        name = company.name
        employees = None # list of {Employee} instances!!!

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
