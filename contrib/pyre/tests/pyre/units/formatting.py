#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the formatting of dimensionals works as expected
"""

def test():
    from pyre.units.SI import meter, second

    g = 9.81*meter/second**2
    accel = 100*g

    assert type(accel/g) == float
    assert accel/g == 100
    assert "{0:value=.2f,base={base},label=g}".format(accel, base=g) == "100.00 g"

    return

# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
