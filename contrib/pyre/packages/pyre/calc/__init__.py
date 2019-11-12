# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package provides the implementation of a simple evaluation network.

There are three fundamental abstractions: variables, operators, and literals. Variables hold
the values computed by the evaluation network, operators compute their values by acting on the
values of other nodes, and literals encapsulate foreign objects, such as numeric constants.
These abstractions provide the machinery for representing arbitrary expressions as graphs.

The interesting aspect of this package is that nodal values get updated automatically when the
values of any of the nodes in their domain change. Nodes keep track of the set of dependents
that are interested in their values and post notifications when their values change.

In addition, this package provides {SymbolTable}, a simple manager for evaluation nodes. Beyond
node storage, {SymbolTable} enables the naming of nodes and can act as the name resolution
context for {Expression} nodes, which evaluate strings with arbitrary python expressions that
may involve the values of other nodes in the model. The other nodes provided here operate
independently of {SymbolTable}. However, it is a good idea to build some kind of container to
hold nodes while the evaluation graph is in use.

Simple examples of the use of the ideas in this package are provided in the unit tests. For a
somewhat more advanced example, take a look at {pyre.config.Configurator}, which is a
{Hierarchical} model that builds an evaluation network out of the traits of pyre components, so
that trait settings can refer to the values of other traits in the configuration files.
"""


# the node generator
from .Calculator import Calculator as calculator


# implementation note: these factories are functions (rather than a raw import of the
# corresponding constructor) in order to prevent the secondary {import} from happening when the
# package itself is first imported. this enables the package to override compile time settings
# and makes it possible to implement the {debug} capability


# factories
# model
def model(**kwds):
    """
    Build a node container that specializes in names that have encoded hierarchical levels,
    such as file paths or namespaces
    """
    from .Hierarchical import Hierarchical
    return Hierarchical(**kwds)


# nodes
def var(value=None, **kwds):
    """
    Build a variable, i.e. a node that can hold an arbitrary value
    """
    # get the base node
    from .Node import Node
    # build a variable and return it
    return Node.variable(value=value, **kwds)


def expression(*, formula, model):
    """
    Build a new node that evaluates a {formula} that involves the names of other nodes as
    resolved in the symbol table {model}.
    """
    # build the node and return it
    return model.expression(value=formula)


def sequence(*operands):
    """
    Build a node that holds a sequence of other nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.sequence(operands=operands)


def mapping(**operands):
    """
    Build a node that holds a sequence of other nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.mapping(operands=operands)


def average(*operands):
    """
    Compute the average of a collection of nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.average(operands=operands)


def count(*operands):
    """
    Compute the length of a collection of nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.count(operands=operands)


def max(*operands):
    """
    Compute the minimum of a collection of nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.max(operands=operands)


def min(*operands):
    """
    Compute the minimum of a collection of nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.min(operands=operands)


def product(*operands):
    """
    Compute the sum of a collection of nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.product(operands=operands)


def sum(*operands):
    """
    Compute the sum of a collection of nodes
    """
    # access the constructor
    from .Node import Node
    # build the node and return it
    return Node.sum(operands=list(operands))


def debug():
    """
    Support for debugging the calc package
    """
    # print(" ++ debugging 'pyre.calc'")
    # attach {ExtentAware} as the metaclass of {Node} so we can verify that all instances of
    # this class are properly garbage collected
    from ..patterns.ExtentAware import ExtentAware
    # get the normal metaclass
    global calculator
    # derive a new one
    class counted(calculator, ExtentAware): pass
    # and set it as the default
    calculator = counted
    # all done
    return


# end of file
