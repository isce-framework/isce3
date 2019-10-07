#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate the sqlite component
"""


def test():
    # access the package
    import pyre.db

    # build a database component
    db = pyre.db.sqlite()

    # and return it
    return db


# main
if __name__ == "__main__":
    test()


# end of file
