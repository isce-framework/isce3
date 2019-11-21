# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre
import pyre.weaver
from . import datastore, sql


# declaration
class Server(pyre.component, implements=datastore):
    """
    Abstract component that encapsulates the connection to a database back end

    This class is meant to be used as the base class for back end specific component
    implementations. It provides a complete but trivial implementation of the {DataStore}
    interface.
    """


    # types
    # exceptions
    from . import exceptions


    # constants
    providesHeaders = True


    # traits
    sql = pyre.weaver.language(default=sql)
    sql.doc = "the generator of the SQL statements"


    # required interface
    @pyre.export
    def attach(self):
        """
        Connect to the database back end
        """
        raise NotImplementedError(
            "class {.__name__!r} must override 'attach'".format(type(self)))

    @pyre.export
    def detach(self):
        """
        Connect to the database back end
        """
        raise NotImplementedError(
            "class {.__name__!r} must override 'attach'".format(type(self)))


    @pyre.export
    def execute(self, *sql):
        """
        Execute the sequence of SQL statements in {sql} as a single command
        """
        raise NotImplementedError(
            "class {.__name__!r} must override 'execute'".format(type(self)))


    # convenience
    def createDatabase(self, name):
        """
        Build and execute the SQL statement to create the database {name}
        """
        # build the sql statement
        sql = self.sql.createDatabase(name=name)
        # and execute it
        return self.execute(*sql)


    def dropDatabase(self, name):
        """
        Build and execute the SQL statement to drop the database {name}
        """
        # build the sql statement
        sql = self.sql.dropDatabase(name=name)
        # and execute it
        return self.execute(*sql)


    def createTable(self, table):
        """
        Build and execute the SQL statement necessary to create {table}
        """
        # build the sql statement
        sql = self.sql.createTable(table)
        # and execute it
        return self.execute(*sql)


    def dropTable(self, table):
        """
        Build and execute the SQL statement necessary to delete {table} from the datastore
        """
        # build the sql statement
        sql = self.sql.dropTable(table)
        # and execute it
        return self.execute(*sql)


    def insert(self, *records):
        """
        Insert {records} into the database
        """
        # if there are no records to insert, bail
        if not records: return
        # build the sql statements
        sql = self.sql.insertRecords(*records)
        # and execute
        return self.execute(*sql)


    def update(self, *specifications):
        """
        Use {specifications} to update the database

        Each item in {specifications} is a pair of a {template} and a {condition}. The
        {template} is an instance of a table row with all fields that require update having
        values that are not {None}. The {condition} is an expression that identifies the
        portion of the table that will be affected
        """
        # go through the {specifications}
        for template, condition in specifications:
            # build the sql statement for this update
            sql = self.sql.updateRecords(template=template, condition=condition)
            # and execute it
            self.execute(*sql)
        # all done
        return


    def delete(self, table, condition):
        """
        Delete all {table} records that match {condition}
        """
        # build the sql statements
        sql = self.sql.deleteRecords(table=table, condition=condition)
        # and execute
        return self.execute(*sql)


    def select(self, query):
        """
        Execute the given {query} and return the retrieved data
        """
        # build the sql statements
        sql = self.sql.select(query=query)
        # execute them
        results = iter(self.execute(*sql))

        # get the headers, if the server provides them; ignore them, for now, since the order
        # of the results matches exactly the field order, by construction
        if self.providesHeaders: headers = next(results)

        # for each row with actual data
        for row in results:
            # build a named tuple
            yield query.pyre_immutable(data=row)
        # all done
        return


    # meta methods
    # context manager support
    def __enter__(self):
        """
        Hook invoked when the context manager is entered
        """
        return self


    def __exit__(self, exc_type, exc_instance, exc_traceback):
        """
        Hook invoked when the context manager's block exits
        """
        # re-raise any exception that occurred while executing the body of the with statement
        return False


# end of file
