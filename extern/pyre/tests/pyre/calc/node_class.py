#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that the refcount is zero after all nodes have gone out of scope
"""


def test():
    # get calc
    import pyre.calc
    # and algebraic
    import pyre.algebraic
    # save the metaclass
    calculator = pyre.calc.calculator
    # and the base node
    base = pyre.algebraic.node

    # make a node class
    class node(metaclass=calculator): pass

    # verify that the {mro} is what we expect
    assert node.__mro__ == (
        node,
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # check literals
    assert node.literal.__mro__ == (
        node.literal, # the leaf
        calculator.const, calculator.observable, calculator.reactor,  # from calculator
        calculator.literal, calculator.leaf, # from algebra
        node, # from node
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # check variables
    assert node.variable.__mro__ == (
        node.variable, # the leaf
        calculator.filter, calculator.memo, # from calculator
        calculator.preprocessor, calculator.postprocessor, # from calculator
        calculator.dependency, calculator.observable, calculator.reactor, # from calculator
        calculator.value, # from calculator
        calculator.variable, calculator.leaf, # from algebra
        node, # from node
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # check operator
    assert node.operator.__mro__ == (
        node.operator, # the leaf
        calculator.memo, # from calculator
        calculator.preprocessor, calculator.postprocessor, # from calculator
        calculator.dependent, calculator.observer, # from calculator
        calculator.dependency, calculator.observable, # from calculator
        calculator.reactor, # from calculator
        calculator.evaluator, # from calculator
        calculator.operator, node.composite, calculator.composite, # from algebra
        node, # from node
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # check expression
    assert node.expression.__mro__ == (
        node.expression, # the leaf
        calculator.memo, # from calculator
        calculator.preprocessor, calculator.postprocessor, # from calculator
        calculator.dependent, calculator.observer, # from calculator
        calculator.dependency, calculator.observable, # from calculator
        calculator.reactor, # from calculator
        calculator.expression, # from calculator
        node.composite, calculator.composite, # from algebra
        node, # from node
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # check interpolation
    assert node.interpolation.__mro__ == (
        node.interpolation, # the leaf
        calculator.memo, # from calculator
        calculator.preprocessor, calculator.postprocessor, # from calculator
        calculator.dependent, calculator.observer, # from calculator
        calculator.dependency, calculator.observable, # from calculator
        calculator.reactor, # from calculator
        calculator.interpolation, # from calculator
        node.composite, calculator.composite, # from algebra
        node, # from node
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # check reference
    assert node.reference.__mro__ == (
        node.reference, # the leaf
        calculator.memo, # from calculator
        calculator.preprocessor, calculator.postprocessor, # from calculator
        calculator.dependent, calculator.observer, # from calculator
        calculator.dependency, calculator.observable, # from calculator
        calculator.reactor, # from calculator
        calculator.reference, # from calculator
        node.composite, calculator.composite, # from algebra
        node, # from node
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # check unresolved nodes
    assert node.unresolved.__mro__ == (
        node.unresolved, # the leaf
        calculator.observable, calculator.reactor, # from calculator
        calculator.unresolved, # from calculator
        calculator.leaf, # from algebra
        node, # from node
        calculator.base, base,
        calculator.arithmetic, calculator.ordering, calculator.boolean,
        object)

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
