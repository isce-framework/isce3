#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Basic AutoNode exercises
"""

def test():
    # access the package
    import pyre.calc

    # make two nodes
    v1 = pyre.calc.var(value=1)
    v2 = pyre.calc.var(value=2)
    # and a couple that reference them
    s = v1 + v2
    p = v1 * v2
    # check
    assert list(s.operands) == [v1, v2]
    assert set(v1.observers) == {s, p}
    assert set(v2.observers) == {s, p}
    # verify that both s and p are dirty
    assert s.dirty == True
    assert p.dirty == True
    # compute the values
    assert s.value == 3
    assert p.value == 2
    # verify that both s and p are now clean
    assert s.dirty == False
    assert p.dirty == False
    # update the value of v1
    v1.value = 2
    # verify that both s and p are dirty again
    assert s.dirty == True
    assert p.dirty == True
    # compute their values
    assert s.value == 4
    assert p.value == 4
    # verify that both s and p are now clean
    assert s.dirty == False
    assert p.dirty == False
    # update the value of v2
    v2.value = 3
    # verify that both s and p are dirty again
    assert s.dirty == True
    assert p.dirty == True
    # compute their values
    assert s.value == 5
    assert p.value == 6
    # verify that both s and p are now clean
    assert s.dirty == False
    assert p.dirty == False

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
