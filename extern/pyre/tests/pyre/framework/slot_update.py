#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise value updates for slots
"""

def test():
    # for the locators
    import pyre.tracking
    # get the constructor
    from pyre.framework.Slot import Slot
    # build a few slots
    var = Slot.variable(key=None, value=None)
    one = Slot.variable(key=None, value=1)
    two = Slot.variable(key=None, value=2)

    three = one + two
    double = var + var

    # check that {var} has no value
    assert var.value is None
    # set it to {two)
    var.value = two.value
    # check that the cache has the right value
    assert var._value == 2
    # and that {var} knows it
    assert var.value == 2
    # and that double got updated
    assert double.value == 2 * var.value

    # verify {three} can compute correctly
    assert three.value == 3

    # now, replace {var} by {three} in {double}
    three.replace(var)

    # check all expected invariants
    # operands
    assert len(tuple(one.operands)) == 0
    assert len(tuple(two.operands)) == 0
    assert len(tuple(var.operands)) == 0
    assert len(tuple(three.operands)) == 2
    assert identical(three.operands, [one, two])
    assert len(tuple(double.operands)) == 2
    assert identical(double.operands, [three, three])

    # observers
    assert len(tuple(one.observers)) == 1
    assert identical(one.observers, [three])
    assert len(tuple(two.observers)) == 1
    assert identical(two.observers, [three])
    assert len(tuple(var.observers)) == 0
    assert len(tuple(three.observers)) == 1
    assert identical(three.observers, [double])
    assert len(tuple(double.observers)) == 0

    # all done
    return var, double


def identical(s1, s2):
    """
    Verify that the nodes in {s1} and {s2} are identical. This has to be done carefully since
    we must avoid triggering __eq__
    """
    # for the pairs
    for n1, n2 in zip(s1, s2):
        # check them for _identity_, not _equality_
        if n1 is not n2: return False
    # all done
    return True


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
