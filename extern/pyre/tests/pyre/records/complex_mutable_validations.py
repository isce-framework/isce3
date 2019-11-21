#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise validators attached to the fields of complex mutable records
"""


def test():
    import pyre.records
    import pyre.constraints

    class record(pyre.records.record):
        """
        A sample record
        """
        sku = pyre.records.str()
        cost = pyre.records.float()

        price = -1.25 * cost
        price.validators = pyre.constraints.isPositive()


    # build a record
    r = record.pyre_mutable(data=("9-4013", "1.0"))

    # attempt to
    try:
        # access the price
        r.price
        # build an invalid record
        assert False
    # it should fail
    except record.ConstraintViolationError as error:
        # check
        assert error.constraint is record.price.validators[0]
        assert error.value == -1.25

    return record


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
