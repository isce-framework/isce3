#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the unit parser works as advertised
"""


def test():
    import pyre.units
    parser = pyre.units.parser()

    import pyre.units.SI as SI

    assert SI.kilogram == parser.parse("kilogram")
    assert SI.meter == parser.parse("meter")
    assert SI.second == parser.parse("second")
    assert SI.ampere == parser.parse("ampere")
    assert SI.mole == parser.parse("mole")
    assert SI.candela == parser.parse("candela")

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
