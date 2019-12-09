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
        number = pyre.properties.int(default=0)
        number.validators = pyre.constraints.isGreater(value=0)

    # and another that assigns the validators in an iterable
    class canonical(pyre.component):
        """the base component"""
        number = pyre.properties.int(default=0)
        number.validators = (pyre.constraints.isGreater(value=0),)

    # check the simple case
    try:
        raw.number
        assert False
    except pyre.component.ConstraintViolationError:
        pass

    # check the iterable case
    try:
        canonical.number()
        assert False
    except pyre.component.ConstraintViolationError:
        pass

    return


# main
if __name__ == "__main__":
    test()


# end of file
