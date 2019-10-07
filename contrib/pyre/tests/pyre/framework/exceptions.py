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

    from pyre.framework.exceptions import (
        FrameworkError, BadResourceLocatorError
        )

    try:
        raise FrameworkError()
    except FrameworkError as error:
        pass

    try:
        raise BadResourceLocatorError(uri=None, reason=None)
    except BadResourceLocatorError as error:
        pass

    return


# main
if __name__ == "__main__":
    test()


# end of file
