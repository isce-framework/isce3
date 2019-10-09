# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class SymbolTable:
    """
    Storage and naming services for algebraic nodes
    """


    # types
    from .Node import Node as node

    # exceptions; included here for client convenience
    from .exceptions import (
        NodeError,
        CircularReferenceError,
        EmptyExpressionError, ExpressionSyntaxError, EvaluationError,
        UnresolvedNodeError
        )


    # public data
    @property
    def keys(self):
        """
        Build an iterator over my registered keys
        """
        # delegate to my map and realize into a container since some types of access modify the
        # key and slot stores
        return tuple(self._nodes.keys())


    @property
    def nodes(self):
        """
        Build an iterator over my registered nodes
        """
        # delegate to my map and realize into a container since some types of access modify the
        # key and slot stores
        return tuple(self._nodes.values())


    # convenience: node constructors
    def expression(self, value, **kwds):
        """
        Build an expression node
        """
        # if the value is already a node
        if isinstance(value, self.node):
            # just return it
            return value
        # if it is not a string
        if not isinstance(value, str):
            # just make a variable
            return self.variable(value=value, **kwds)
        # otherwise, attempt to
        try:
            # compile the {value}
            program, operands = self.node.expression.compile(model=self, expression=value)
        # if this fails
        except self.node.EmptyExpressionError as error:
            # make a variable instead; use the processed value to get rid of the meta-characters
            return self.variable(value=error.expression, **kwds)
        # otherwise, build an expression
        return self.node.expression(
            model=self, expression=value, program=program, operands=operands, **kwds)


    def interpolation(self, value, **kwds):
        """
        Build an interpolation node
        """
        # if the value is already a node
        if isinstance(value, self.node):
            # just return it
            return value
        # if it is not a string
        if not isinstance(value, str):
            # just make a variable
            return self.variable(value=value, **kwds)
        # otherwise, attempt to
        try:
            # compile the {value}
            operands = self.node.interpolation.compile(model=self, expression=value)
        # if this fails
        except self.node.EmptyExpressionError as error:
            # make a variable instead; use the processed value to get rid of the meta-characters
            return self.variable(value=error.expression, **kwds)
        # otherwise, build an interpolation
        return self.node.interpolation(model=self, expression=value, operands=operands, **kwds)


    def sequence(self, nodes, **kwds):
        """
        Build a sequence node
        """
        # easy enough
        return self.node.sequence(operands=tuple(nodes), **kwds)


    def mapping(self, nodes, **kwds):
        """
        Build a mapping node
        """
        # easy enough
        return self.node.mapping(operands=nodes, **kwds)


    def variable(self, **kwds):
        """
        Build a variable
        """
        # easy enough
        return self.node.variable(**kwds)


    def literal(self, **kwds):
        """
        Build a literal node
        """
        # easy enough
        return self.node.literal(**kwds)


    # interface
    def get(self, name, default=None):
        """
        Attempt to resolve {name} and return its value; if {name} is not in the symbol table,
        bind it to {default} and return this new value
        """
        # attempt to evaluate
        try:
            return self[name]
        # if it failed
        except self.UnresolvedNodeError:
            # bind it to {default}
            self[name] = default
        # evaluate and return
        return self[name]


    def insert(self, name, node):
        """
        Insert {node} in the symbol table under {name}. If there is an existing node, adjust the
        evaluation graph by replacing the existing node with the new one everywhere
        """
        # look for
        try:
            # an existing one
            old = self._nodes[name]
        # if not there
        except KeyError:
            # no problem
            pass
        # if there is an existing node
        else:
            # replace it
            node.replace(old)

        # update the model
        self._nodes[name] = node

        # and return
        return


    def retrieve(self, name):
        """
        Retrieve the node registered under {name}. If no such node exists, an error marker will
        be built, stored in the symbol table under {name}, and returned.
        """
        # if a node is already registered under {name}
        try:
            # grab it
            node = self._nodes[name]
        # otherwise
        except KeyError:
            # build an error marker
            node = self.node.unresolved(request=name)
            # add it to the pile
            self._nodes[name] = node
        # and return it
        return node


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my node map
        self._nodes = {}
        # all done
        return


    def __contains__(self, name):
        """
        Check whether {name} is present in the symbol table
        """
        # check whether it is present in my node index
        return name in self._nodes


    def __iter__(self):
        """
        Go through my keys
        """
        # easy enough
        return iter(self._nodes)


    def __getitem__(self, name):
        """
        Look up the node registered under {name} and return its value
        """
        # get the node
        node = self.retrieve(name)
        # compute and return its value
        return node.value


    def __setitem__(self, name, value):
        """
        Add or update the named node with the given {value}
        """
        # convert {value} into an appropriate node
        new = self.interpolation(value=value)
        # insert it into the model and return
        return self.insert(node=new, name=name)


    # private data
    _nodes = None


# end of file
