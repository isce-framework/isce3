#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that manipulating the default and instance trait values has no cross-talk
"""

import pyre

class gaussian(pyre.component):
    """a representation of a gaussian function"""

    # properties
    mean = pyre.properties.float(default=0)
    spread = pyre.properties.float(default=.25)


def test():

    # first, the basics
    assert gaussian.mean == 0
    assert gaussian.spread == 0.25
    # change the defaults
    gaussian.mean = 1
    gaussian.spread = 0.5
    # and check again
    assert gaussian.mean == 1
    assert gaussian.spread == 0.5

    # instantiate a couple of functors
    g1 = gaussian(name='g1')
    g2 = gaussian(name='g2')
    # check that their traits gave the default values
    assert g1.mean == 1
    assert g1.spread == .5
    assert g2.mean == 1
    assert g2.spread == .5

    # now, change the traits values of g1
    g1.mean = 0
    g1.spread = 0.25
    # verify that everybody is configured correctly
    assert gaussian.mean == 1
    assert gaussian.spread == 0.5
    assert g1.mean == 0
    assert g1.spread == .25
    assert g2.mean == 1
    assert g2.spread == .5

    # finally, change the default values again
    gaussian.mean = 2
    gaussian.spread = 0.75
    # and verify that everybody is still ok
    assert gaussian.mean == 2
    assert gaussian.spread == 0.75
    assert g1.mean == 0
    assert g1.spread == .25
    assert g2.mean == 2
    assert g2.spread == .75

    return gaussian, g1, g2


# main
if __name__ == "__main__":
    test()


# end of file
