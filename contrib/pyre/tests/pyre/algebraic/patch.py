#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we can traverse the expression tree correctly and completely
"""


def test():
    # access to the package
    import pyre.algebraic

    # declare a node class
    class node(metaclass=pyre.algebraic.algebra): pass

    # declare a couple of nodes
    n1 = node.variable()
    n2 = node.variable()
    n3 = node.variable()

    # careful with comparisons: do not trigger operator _eq_!
    # check that they have no dependencies
    assert list(map(id, n1.variables)) == [id(n1)]
    assert list(map(id, n2.variables)) == [id(n2)]
    assert list(map(id, n3.variables)) == [id(n3)]

    # a simple expression
    n = n1 + n2
    assert list(map(id, n.variables)) == [id(n1), id(n2)]
    # patch {n3} in
    n.substitute(current=n1, replacement=n3)
    # and check that it happened correctly
    assert list(map(id, n.variables)) == [id(n3), id(n2)]

    # add another layer
    m = n1 + n2 + n3
    assert list(map(id, m.variables)) == [id(n1), id(n2), id(n3)]
    # patch {n} in
    m.substitute(current=n3, replacement=n)
    # check
    assert list(map(id, m.variables)) == [id(n1), id(n2), id(n3), id(n2)]

    # a more complicated example
    n = (2*(n1**2 - 2*n1*n2 + n2**2)*n3)
    assert set(map(id, n.variables)) == {id(n1), id(n2), id(n3)}
    # patch {n3} in
    n.substitute(current=n1, replacement=n3)
    # and check that it happened correctly
    assert set(map(id, n.variables)) == {id(n2), id(n3)}

    # let's try to make a cycle
    try:
        n.substitute(current=n2, replacement=n)
        assert False
    except n.CircularReferenceError:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
