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
    import pyre.calc
    # declare a couple of nodes
    n1 = pyre.calc.var(value=1)
    n2 = pyre.calc.var(value=2)
    # unary operators
    assert (- n1).value == -1
    assert (+ n2).value == 2
    assert (abs(n1)).value == 1
    # basic arithmetic with two operands
    assert (n1 + n2).value == 1 + 2
    assert (n1 - n2).value == 1 - 2
    assert (n1 * n2).value == 1 * 2
    assert (n1 / n2).value == 1 / 2
    # assert (n1 // n2).value == 1 // 2 # NYI
    # basic arithmetic with more than two operands
    assert (n1 + n2 - n1).value == 1 + 2 - 1
    assert (n1 * n2 / n1).value == 1 * 2 / 1
    assert ((n1 - n2)*n2).value == (1 - 2)*2
    # basic arithmetic with floats
    assert (1 + n2).value == 1 + 2
    assert (n2 + 1).value == 2 + 1
    assert (1 - n2).value == 1 - 2
    assert (n2 - 1).value == 2 - 1
    assert (2 * n1).value == 2 * 1
    assert (n1 * 2).value == 1 * 2
    assert (3 / n2).value == 3 / 2
    assert (n2 / 3).value == 2 / 3
    assert (n2 ** 3).value == 2**3
    assert (3 ** n2).value == 3**2

    # more complicated forms
    assert ((n1**2 + 2*n1*n2 + n2**2)).value == ((n1+n2)**2).value
    assert ((n1**2 - 2*n1*n2 + n2**2)).value == ((n1-n2)**2).value
    assert (2*(.5 - n1*n2 + n2**2)*n1).value == 2*(.5 - 1*2 + 2**2)*1

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
