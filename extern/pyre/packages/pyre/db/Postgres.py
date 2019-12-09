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
class Postgres(Server, family="pyre.db.server.postgres"):
    """
    Component that manages the connection to a Postgres database
    """

    # exceptions
    from pyre.db.exceptions import OperationalError


    # public state
    database = pyre.properties.str(default="postgres")
    database.doc = "the name of the database to connect to"

    username = pyre.properties.str(default=None)
    username.doc = "the database user name to use during authentication"

    password = pyre.properties.str(default=None)
    password.doc = "the password of the database user"

    application = pyre.properties.str(default=None)
    application.doc = "the application name to use for the connection"

    quiet = pyre.properties.bool(default=True)
    quiet.doc = "control whether certain postgres informationals are shown"


    # interface
    @pyre.export
    def attach(self):
        """
        Connect to the database
        """
        # if i have an existing connection to the back end, do nothing
        if self.connection is not None: return

        # otherwise, build the connection specification string
        spec = [
            # the name of the database is required
            ['dbname', self.database]
            ]
        # the others are optional, depending on how the database is configured
        if self.username is not None: spec.append(['user', self.username])
        if self.password is not None: spec.append(('password', self.password))
        if self.application is not None: spec.append(('application_name', self.application))
        # put it all together
        spec = ' '.join('='.join(entry) for entry in spec)

        # establish the connection
        self.connection = self.postgres.connect(spec)

        # if the user asked for {quiet} operation
        if self.quiet:
            # set the minimum diagnostic level to {warning}
            self.execute("SET client_min_messages = warning;")

        # all done
        return self


    @pyre.export
    def detach(self):
        """
        Close the connection to the database

        Closing a connection makes it unsuitable for any further database access. This applies
        to all objects that may retain a reference to the connection being closed. Any
        uncommitted changes will be lost
        """
        # if i don't have an existing connection to the back end, do nothing
        if self.connection is None: return

        # otherwise, close the connection
        status = self.postgres.disconnect(self.connection)
        # invalidate the member
        self.connection = None

        # and return the status
        return status


    @pyre.export
    def execute(self, *sql):
        """
        Execute the sequence of SQL statements in {sql} as a single command
        """
        # assemble the command and pass it on to the connection
        return self.postgres.execute(self.connection, "\n".join(sql))


    # meta methods
    def __new__(cls, **kwds):
        # if necessary
        if cls.postgres is None:
            # initialize the extension module
            cls.postgres = cls.initializeExtension()
        # chain up
        return super().__new__(cls, **kwds)


    # context manager interface
    def __enter__(self):
        """
        Hook invoked when the context manager is entered
        """
        # mark the beginning of a transaction
        self.execute(*self.sql.transaction())
        # and hand me back to the caller
        return self


    def __exit__(self, exc_type, exc_instance, exc_traceback):
        """
        Hook invoked when the context manager's block exits
        """
        # if there were no errors detected
        if exc_type is None:
            # commit the transaction to the datastore
            self.execute(*self.sql.commit())
        # otherwise
        else:
            # roll back
            self.execute(*self.sql.rollback())

        # indicate that we want to re-raise any exceptions that occurred while executing the
        # body of the {with} statement
        return False


    # implementation details
    postgres = None # the handle to the extension module
    connection = None # the handle to the session with the back-end


    # helper routine to initialize the extension module
    @classmethod
    def initializeExtension(cls):
        # access the extension
        from ..extensions import postgres

        # pull in the {NULL} object rep
        from . import null
        # register it with the extension
        postgres.registerNULL(null)

        # get hold of the standard compliant exception hierarchy
        from . import exceptions
        # register the exception hierarchy with the module so that the exceptions it raises are
        # subclasses of the ones defined in pyre.db
        postgres.registerExceptions(exceptions)
        # and return the module
        return postgres


# end of file
