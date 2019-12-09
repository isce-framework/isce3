# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Composite:
    """
    Mix-in class that provides an implementation of the subset of the interface of {Node} that
    requires traversal of the expression graph rooted at nodes with dependencies.

    This class assumes that its instances provide {operands}, a tuple of their dependencies on
    other nodes
    """


    # types
    from .exceptions import CircularReferenceError


    # interface
    @property
    def operands(self):
        """
        A sequence of my direct dependents
        """
        # the default implementation stores my operands in a private member
        return self._operands


    @property
    def span(self):
        """
        Return a sequence over my entire dependency graph
        """
        # i am a node in my dependency graph
        yield self
        # go through my operands
        for operand in self.operands:
            # and ask them for their span
            yield from operand.span
        # all done
        return


    # classifiers
    @property
    def literals(self):
        """
        Return a sequence over the nodes in my dependency graph that encapsulate foreign objects
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for literals in their span
            yield from operand.literals
        # all done
        return


    @property
    def operators(self):
        """
        Return a sequence over the composite nodes in my dependency graph
        """
        # i am one
        yield self
        # go through my operands
        for operand in self.operands:
            # and ask them for operators in their span
            yield from operand.operators
        # all done
        return


    @property
    def variables(self):
        """
        Return a sequence over the leaf nodes in my dependency graph
        """
        # go through my operands
        for operand in self.operands:
            # and ask them for variables in their span
            yield from operand.variables
        # all done
        return


    # alterations to the dependency graph
    def substitute(self, current, replacement, clean=None):
        """
        Traverse my span and replace all occurrences of {current} with {replacement}

        This method makes it possible to introduce cycles in the dependency graph, which is not
        desirable typically. To prevent this, we check that {self} is not in the span of
        {replacement} when the caller does not supply a set of {clean} nodes
        """
        # if the caller didn't hand me a {clean} pile
        if clean is None: clean = set()
        # if the {clean} pile does not already contain {replacement}
        if replacement not in clean:
            # cycle detection: look for {self} in the span of {replacement}; do it carefully so
            # as not to trigger a call to the potentially overloaded {__eq__}, which may not
            # actually perform a comparison
            for node in replacement.span:
                # is this a match
                if node is self:
                    # the substitution would create a cycle
                    raise self.CircularReferenceError(node=self)
            # all good; put {replacement} in the pile of {clean} nodes
            clean.add(replacement)
        # now, iterate over composites in my span
        for node in self.operators:
            # if this is a node we have visited before
            if node in clean:
                # skip it
                continue
            # otherwise, perform the substitution
            node._substitute(current=current, replacement=replacement)
            # and mark this node as clean
            clean.add(node)

        # all done
        return clean


    # meta-methods
    def __init__(self, operands, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my direct dependencies
        self._operands = tuple(operands)
        # all done
        return


    # implementation details
    def _substitute(self, current, replacement):
        """
        Adjust the operands by substituting {replacement} for {current} in the sequence of operands
        """
        # my new pile of operands
        operands = tuple(
            # consists of replacing {current} with {replacement} wherever i bump into it
            replacement if operand is current else operand
            # in the pile of dependencies
            for operand in self.operands
        )
        # attach the new pile
        self._operands = operands
        # all done
        return self


# end of file
