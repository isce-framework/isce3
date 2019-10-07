#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise "isPositive"
"""


def test():
    import pyre.constraints
    constrain = pyre.constraints.isPositive()

    constrain(0)
    constrain(1)
    constrain(1.1)
    constrain(2)

    stranger = -1
    try:
        constrain(stranger)
    except constrain.ConstraintViolationError as error:
        assert error.constraint == constrain
        assert error.value == stranger

    return constrain


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
