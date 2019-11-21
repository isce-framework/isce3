# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Composite:
    """
    Mix-in class that augments raph traversal for the new leaves defined in this package
    """


    # classifiers
    @property
    def expressions(self):
        """
        Return a sequence over the nodes in my dependency graph that are constructed out of python
        expressions involving the names of other nodes
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for variables in their span
            yield from operand.expressions
        # all done
        return


    @property
    def interpolations(self):
        """
        Return a sequence over the nodes in my dependency graph that are constructed by expanding
        the values of other nodes in a macro
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for variables in their span
            yield from operand.expressions
        # all done
        return


    @property
    def mappings(self):
        """
        Return a sequence over the nodes in my dependency graph that are mappings
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for mappings in their span
            yield from operand.mappings
        # all done
        return


    @property
    def references(self):
        """
        Return a sequence over the nodes in my dependency graph that are references to other nodes
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for references in their span
            yield from operand.references
        # all done
        return


    @property
    def sequences(self):
        """
        Return a sequence over the nodes in my dependency graph that are sequences
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for sequences in their span
            yield from operand.sequences
        # all done
        return


    @property
    def uresolveds(self):
        """
        Return a sequence over the unresolved nodes in my dependency graph
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for variables in their span
            yield from operand.unresolveds
        # all done
        return


# end of file
