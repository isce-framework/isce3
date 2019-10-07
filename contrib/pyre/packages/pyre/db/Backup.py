# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
# superclass
from .Server import Server


# declaration
class Backup(Server, family="pyre.db.server.backup"):
    """
    Component that saves SQL statement in a stream
    """

    # public state
    database = pyre.properties.str(default="what?")
    database.doc = "the name of the database to connect to"

    stream = pyre.properties.ostream(default="stdout")
    stream.doc = "the stream in which to place SQL statements"


    # interface
    @pyre.export
    def attach(self):
        """
        Connect to the database
        """
        # nothing to do
        return


    @pyre.export
    def detach(self):
        """
        Close the connection to the database
        """
        # nothing to do
        return


    @pyre.export
    def execute(self, *sql):
        """
        Execute the sequence of SQL statements in {sql} as a single command
        """
        # assemble the SQL statements
        statements = '\n'.join(sql)
        # place them in my stream
        print(statements, file=self.stream)
        # all done
        return


    # meta methods
    # context manager interface
    def __enter__(self):
        """
        Hook invoked when the context manager is entered
        """
        # nothing to do
        return self


    def __exit__(self, exc_type, exc_instance, exc_traceback):
        """
        Hook invoked when the context manager's block exits
        """
        # nothing to do
        return


# end of file
