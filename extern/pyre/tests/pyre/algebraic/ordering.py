#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise node algebra
"""


def test():
    # access the various operator
    import operator
    # access the package
    import pyre.algebraic

    # declare a node class
    class node(metaclass=pyre.algebraic.algebra): pass

    # declare a couple of nodes
    n1 = node.variable()
    n2 = node.variable()

    # check some expressions
    check_binary(n1 == n2, operator.eq, n1, n2)
    check_binary(n1 <= n2, operator.le, n1, n2)
    check_binary(n1 < n2, operator.lt, n1, n2)
    check_binary(n1 >= n2, operator.ge, n1, n2)
    check_binary(n1 > n2, operator.gt, n1, n2)
    check_binary(n1 != n2, operator.ne, n1, n2)

    # all done
    return node


# the checker
def check_binary(expression, operator, op1, op2):
    assert expression.evaluator is operator
    assert expression._operands[0] is op1
    assert expression._operands[1] is op2
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
