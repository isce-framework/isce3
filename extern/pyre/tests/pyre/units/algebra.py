#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

def test():
    import math
    import pyre.units

    m = pyre.units.dimensional(value=1, derivation=(1,0,0,0,0,0,0))
    r = pyre.units.dimensional(value=2, derivation=(0,1,0,0,0,0,0))
    t = pyre.units.dimensional(value=3, derivation=(0,0,1,0,0,0,0))
    ω = pyre.units.dimensional(value=4, derivation=(0,0,-1,0,0,0,0))

    # positive tests
    # addition
    assert (m+m).value == 2*m.value
    assert (m+m).derivation == m.derivation
    # subtraction
    assert (m-m).value == 0
    assert (m-m).derivation == m.derivation
    # multiplication/division
    assert (m*2).value == 2*m.value
    assert (m*2).derivation == m.derivation
    assert (m/2).value == m.value/2
    assert (m/2).derivation == m.derivation
    p = m*r/t
    assert p.value == 1*2/3
    assert p.derivation == (1,1,-1,0,0,0,0)
    # power
    assert (t**2).value == (t.value)**2
    assert (t**2).derivation == (0,0,2,0,0,0,0)
    # unary minus
    assert (-m).value == -m.value
    assert (-m).derivation == m.derivation
    # abs
    assert abs(-m).value == m.value
    assert abs(-m).derivation == m.derivation
    # right multiplication
    assert (2*m).value == 2*m.value
    assert (2*m).derivation == m.derivation
    # right division
    assert (2/m).value == 2/m.value
    assert (2/m).derivation == (-1,0,0,0,0,0,0)

    # float casts
    assert float(m/m) == 1
    # auto-cancellations
    assert (m/m) == 1
    assert m*(1/m) == 1

    # compatibility
    assert m.isCompatible(2*m)

    # negative tests
    # addition of incompatible types
    try:
        m + 2
        assert False
    except TypeError:
        pass
    # addition of incompatible units
    try:
        m + t
        assert False
    except m.CompatibilityError:
        pass
    # subtraction of incompatible types
    try:
        m - 2
        assert False
    except TypeError:
        pass
    # subtraction of incompatible units
    try:
        m - t
        assert False
    except m.CompatibilityError:
        pass
    # exponentiation
    try:
        m ** m
        assert False
    except TypeError:
        pass

    return

# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# version
__id__ = "$Id$"

# end of file
