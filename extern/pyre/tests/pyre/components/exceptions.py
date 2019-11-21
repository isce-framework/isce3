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
    import pyre
    from pyre.components.exceptions import (
        ComponentError,
        CategoryMismatchError, ImplementationSpecificationError, ProtocolError,
        TraitNotFoundError)

    class component(pyre.component): pass
    class protocol(pyre.protocol): pass

    c1 = component(name="c1")
    c2 = component(name="c2")

    try:
        raise ComponentError()
    except ComponentError as error:
        pass

    try:
        raise CategoryMismatchError(configurable=c1, target=c2, name="test")
    except CategoryMismatchError as error:
        pass

    try:
        raise ImplementationSpecificationError(name="test", errors=[])
    except ImplementationSpecificationError as error:
        pass

    try:
        raise ProtocolError(component=c1, protocol=protocol, report=None)
    except ProtocolError as error:
        pass

    try:
        raise TraitNotFoundError(configurable=c1, name=None)
    except TraitNotFoundError as error:
        pass

    return


# main
if __name__ == "__main__":
    test()


# end of file
