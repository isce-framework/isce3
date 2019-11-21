#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that node substitution respects the expression graph invariants
"""


def test():
    # get the package
    import pyre.calc
    # build some nodes
    n1 = pyre.calc.var()
    n2 = pyre.calc.var()
    n3 = pyre.calc.var()

    # first, something simple
    s = n1 + n2

    # we expect:
    # n1 to have one observer: s
    assert len(tuple(n1.observers)) == 1
    assert identical(n1.observers, [s])
    # n2 to have one observer: s
    assert len(tuple(n2.observers)) == 1
    assert identical(n2.observers, [s])
    # n3 to have no observers
    assert len(tuple(n3.observers)) == 0
    # s to have two operands: n1 and n2
    assert len(tuple(s.operands)) == 2
    assert identical(s.operands, (n1, n2))
    # and no observers
    assert len(tuple(s.observers)) == 0

    # make a substitution
    s.substitute(current=n1, replacement=n3)
    # we expect:
    # n1 to still have one observer, since {substitute} no longer adjusts observer piles
    assert len(tuple(n1.observers)) == 1
    assert identical(n1.observers, [s])
    # n2 to have one observer: s
    assert len(tuple(n2.observers)) == 1
    assert identical(n2.observers, [s])
    # n3 to have no observes (see above)
    assert len(tuple(n3.observers)) == 0
    # s to have two operands: n2 and n3
    assert len(tuple(s.operands)) == 2
    assert identical(s.operands, (n3, n2))
    # and no observers
    assert len(tuple(s.observers)) == 0

    # attempt to create a cycle
    try:
        s.substitute(current=n3, replacement=s)
        assert False
    except s.CircularReferenceError:
        pass

    return


def identical(s1, s2):
    """
    Verify that the nodes in {s1} and {s2} are identical. This has to be done carefully since
    we must avoid triggering __eq__
    """
    # realize both of them
    s1 = tuple(s1)
    s2 = tuple(s2)
    # fail if their lengths are not the same
    if len(s1) != len(s2): return False
    # go through one
    for n1 in s1:
        # and the other
        for n2 in s2:
            # if this is a match, we are done
            if n1 is n2: break
        # if we didn't n1 in s2
        else:
            # fail
            return False
    # if we get this far, all is good
    return True


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
