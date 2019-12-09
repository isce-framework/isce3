#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that improperly formed input source raise predicatble exceptions
"""


def test():

    import pyre.records
    csv = pyre.records.csv()

    # try to read without providing a valid data source
    try:
        tuple(csv.read(layout=pyre.records.record))
        assert False
    except csv.SourceSpecificationError as error:
        assert error.description == "invalid input source specification"

    return csv


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
