# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre


# declaration
class DataStore(pyre.protocol, family="pyre.db.server"):
    """
    Protocol declaration for database managers
    """


    # interface
    @pyre.provides
    def attach(self):
        """
        Establish a connection to the data store
        """

    @pyre.provides
    def detach(self):
        """
        Close a connection to the data store
        """

    @pyre.provides
    def execute(self, *sql):
        """
        Execute the sequence of SQL statements in {sql} as a single command
        """


# end of file
