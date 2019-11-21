#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Test for all the exceptions raised by this package
"""


def test():

    from pyre.config.exceptions import CodecError, DecodingError, EncodingError

    try:
        raise CodecError(codec=None, uri=None, locator=None)
    except CodecError as error:
        pass

    try:
        raise DecodingError(codec=None, uri=None, locator=None)
    except DecodingError as error:
        pass

    try:
        raise EncodingError(codec=None, uri=None, locator=None)
    except EncodingError as error:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
