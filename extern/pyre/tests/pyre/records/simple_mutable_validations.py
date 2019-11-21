#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise validators attached to the fields of simple mutable records
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


    # build an invalid record; construction should succeed, access should fail
    invalid = interval.pyre_mutable(left=1, right=-1)

    # try to access the fields
    try:
        invalid.left
        assert False
    except interval.ConstraintViolationError as error:
        assert error.constraint is interval.left.validators[0]
        assert error.value == 1

    # and again
    try:
        invalid.right
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
