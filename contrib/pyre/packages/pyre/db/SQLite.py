# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
import sqlite3
# superclass
from .Server import Server


# declaration
class SQLite(Server, family="pyre.db.server.sqlite"):
    """
    Component that manages the connection to a sqlite database
    """


    # constants
    providesHeaders = False # sqlite queries do not return column headers


    # public state
    database = pyre.properties.str(default=":memory:")
    database.doc = "the path to the sqlite database"


    # interface
    @pyre.export
    def attach(self):
        """
        Connect to the database
        """
        # if i have an existing connection to the database, do nothing
        if self.connection is not None: return
        # otherwise, make a connection
        self.connection = sqlite3.connect(self.database)
        # and a cursor
        self.cursor = self.connection.cursor()
        # and return
        return self


    @pyre.export
    def detach(self):
        """
        Close the connection to the database
        """
        # if i don't have an existing connection to the database, do nothing
        if self.connection is None: return
        # otherwise, close my cursor
        self.cursor.close()
        # and the connection
        self.connection.close()
        # reset the connection objects
        self.cursor = None
        self.connection = None
        # all done
        return


    @pyre.export
    def execute(self, *sql):
        """
        Execute the sequence of SQL statements in {sql} as a single command
        """
        # splice the statements together and hand them to my cursor
        self.cursor.execute('\n'.join(sql))
        # return the cursor
        return self.cursor


    # implementation details
    cursor = None
    connection = None

# end of file
