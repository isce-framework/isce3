#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that trait assignments that violate constraints are flagged
"""


def test():
    import pyre

    # declare a component
    class base(pyre.component):
        """the base component"""
        positive = pyre.properties.int(default=0)
        positive.validators = (pyre.constraints.isGreaterEqual(value=0), )

        interval = pyre.properties.int(default=0)
        interval.validators = (
            pyre.constraints.isGreaterEqual(value=-1), pyre.constraints.isLess(value=1))

    return

    # instantiate
    b = base(name="b")
    # attempt to
    try:
        # make an assignment that violates the constraint
        b.positive = -1
        # should be unreachable
        assert False
    # if it were detected correctly
    except b.ConstraintViolationError:
        # verify the state didn't change
        assert b.positive == 0

    # attempt to
    try:
        # make another assignment that violates the constraint
        b.interval = 1
        # should be unreachable
        assert False
    # if it were detected correctly
    except b.ConstraintViolationError:
        # verify the state didn't change
        assert b.positive == 0

    return


# main
if __name__ == "__main__":
    test()


# end of file
