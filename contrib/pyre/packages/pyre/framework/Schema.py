# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Schema:
    """
    The singleton that indexes the database schema of a pyre application
    """


    # public data
    models = None # a map of tables to objects that model them


    # interface
    def dependencies(self):
        """
        Build a graph of tables and their dependencies
        """
        # for each registered table
        for table in self.tables:
            # collect the referenced tables
            references = (m._foreign.reference.table for m in table.pyre_measures if m._foreign)
            # yield the table and a tuple of its references; don't be tempted to return the
            # generator as it gets exhausted after the first time it is accessed, which might
            # surprise our clients
            yield table, tuple(references)
        # all done
        return


    def sort(self):
        """
        Visit tables in topological order
        """
        # initialize the pile of previously encountered tables
        done = set()
        # go through all the tables
        for table in self.tables:
            # and sort them
            yield from self._sort(table=table, done=done)
        # all done
        return


    # meta-methods
    def __init__(self, executive, **kwds):
        # chain up
        super().__init__(**kwds)

        # my model index
        self.models = {}
        # the table container
        self.tables = set()

        # all done
        return


    # implementation details
    def _sort(self, table, done):
        """
        The engine of the topological sort
        """
        # if this table has been encountered before
        if table in done:
            # nothing else to do
            return
        # otherwise, add it to the pile
        done.add(table)
        # go through its fields
        for measure in table.pyre_measures:
            # looking only for foreign keys
            if not measure._foreign: continue
            # got one; find the table it refers to
            referent = measure._foreign.table
            # and visit it
            yield from self._sort(table=referent, done=done)
        # its turn
        yield table
        # all done
        return


# end of file
