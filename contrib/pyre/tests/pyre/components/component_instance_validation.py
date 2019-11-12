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

    # instantiate
    b = base(name="b")
    # make an assignment that violates the constraint
    b.positive = -1
    # read the value
    try:
        b.positive
        assert False
    except b.ConstraintViolationError:
        pass

    # and another
    b.interval = 1
    try:
        b.interval
        assert False
    except b.ConstraintViolationError:
        pass

    return


# main
if __name__ == "__main__":
    test()


# end of file
