#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise validators attached to fields of simple immutable records
"""


def test():
    import pyre.records
    import pyre.constraints

    class interval(pyre.records.record):
        """
        A sample record
        """
        # field declarations
        left = pyre.records.float()
        right = pyre.records.float()

        # constraints
        left.validators = pyre.constraints.isLess(value=0)
        right.validators = pyre.constraints.isGreater(value=0)


    # try to
    try:
        # build an invalid record
        interval.pyre_immutable(left=1, right=1)
        assert False
    # it should fail
    except interval.ConstraintViolationError as error:
        # check
        assert error.constraint is interval.left.validators[0]
        assert error.value == 1

    # and again
    try:
        interval.pyre_immutable(left=-1, right=-1)
        assert False
    except interval.ConstraintViolationError as error:
        assert error.constraint is interval.right.validators[0]
        assert error.value == -1

    return interval


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
