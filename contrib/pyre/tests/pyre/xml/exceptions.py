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

    from pyre.xml.exceptions import (
        ParsingError, UnsupportedFeatureError, DTDError, ProcessingError
        )

    try:
        raise ParsingError()
    except ParsingError as error:
        pass

    try:
        raise UnsupportedFeatureError(features=[])
    except UnsupportedFeatureError as error:
        pass

    try:
        raise DTDError()
    except DTDError as error:
        pass

    try:
        raise ProcessingError(saxlocator=None)
    except ProcessingError as error:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
