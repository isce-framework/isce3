#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise "isAll"
"""


def test():
    import pyre.constraints
    less = pyre.constraints.isLess(value=1)
    greater = pyre.constraints.isGreater(value=0)
    constraint = pyre.constraints.isAll(less, greater)

    constraint.validate(0.1)
    constraint.validate(0.5)
    constraint.validate(0.9)

    stranger = 1
    try:
        constraint.validate(stranger)
    except constraint.ConstraintViolationError as error:
        assert error.constraint in [less, greater]
        assert error.value == stranger

    stranger = 0
    try:
        constraint.validate(stranger)
    except constraint.ConstraintViolationError as error:
        assert error.constraint in [less, greater]
        assert error.value == stranger

    return constraint


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
