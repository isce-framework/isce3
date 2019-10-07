#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Tests for all the exceptions raised by this package
"""

def test():

    from pyre.units.exceptions import (
        UnitError, ConversionError, CompatibilityError
        )

    try:
        raise UnitError()
    except UnitError as error:
        pass

    try:
        raise ConversionError(operand=None)
    except ConversionError as error:
        pass

    try:
        raise CompatibilityError(operation=None, op1=None, op2=None)
    except CompatibilityError as error:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
