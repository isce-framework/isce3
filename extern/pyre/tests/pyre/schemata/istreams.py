#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the URI parser
"""


def test():
    import pyre.schemata

    # make a converter
    istream = pyre.schemata.istream()

    # open a file that exists
    f = istream.coerce('file:istreams.py')
    assert f.name == 'istreams.py'

    # a poorly formed one
    try:
        istream.coerce("&")
        assert False
    except istream.CastingError as error:
        assert str(error) == 'unrecognizable URI {!r}'.format('&')

    # anything else?
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
