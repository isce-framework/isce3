#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the trait defaults get validated properly
"""


def test():
    import pyre

    # declare a component
    class raw(pyre.component):
        """the base component"""
        number = pyre.properties.int(default=1)
        number.validators = pyre.constraints.isGreater(value=0)

    # and another that assigns the validators in an iterable
    class canonical(pyre.component):
        """the base component"""
        number = pyre.properties.int(default=1)
        number.validators = (pyre.constraints.isGreater(value=0),)

    return

    # check the simple case
    try:
        # illegal assignment
        raw.number = 0
        # should be unreachable
        assert False
    # if it got detected properly
    except pyre.component.ConstraintViolationError:
        # verify the state is unchanged
        assert raw.number == 1

    # check the iterable case
    try:
        # illegal assignment
        canonical.number = 0
        # should be unreachable
        assert False
    # if it got detected properly
    except pyre.component.ConstraintViolationError:
        # verify the state is unchanged
        assert canonical.number == 1

    return


# main
if __name__ == "__main__":
    test()


# end of file
