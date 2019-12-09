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

    # access the postgres package
    import pyre.db

    # build a database component
    db = pyre.db.postgres()
    # connect to the default database
    db.attach()

    # and return the component
    return db


# main
if __name__ == "__main__":
    test()


# end of file
