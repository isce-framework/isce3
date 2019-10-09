#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Make a connection to a database using the low level interface from the extension module
"""


def test():
    from pyre.extensions import postgres as pyrepg
    # initialize the module exceptions
    import pyre.db.exceptions as exceptions
    pyrepg.registerExceptions(exceptions)

    # make a connection
    connection = pyrepg.connect("dbname=postgres")
    # execute a command
    try:
        pyrepg.execute(connection, "no-such-command")
        assert False
    except pyrepg.ProgrammingError:
        pass

    # and return it
    return connection


# main
if __name__ == "__main__":
    test()


# end of file
