#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise a Fortran77 weaver
"""


def test():
    # get the package
    import pyre.weaver
    # instantiate a weaver
    weaver = pyre.weaver.weaver(name="sanity")
    weaver.language = "f77"

    text = list(weaver.weave())
    assert text == [
        'c -*- Fortran -*-',
        'c',
        'c Michael A.G. Aïvázis',
        'c Orthologue',
        'c (c) 1998-2019 All Rights Reserved',
        'c',
        '',
        '',
        'c end of file',
        ]

    return


# main
if __name__ == "__main__":
    test()


# end of file
