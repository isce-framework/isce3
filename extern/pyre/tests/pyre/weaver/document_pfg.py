#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise a PFG weaver
"""


def test():
    # get the package
    import pyre.weaver
    # instantiate a weaver
    weaver = pyre.weaver.weaver(name="sanity")
    weaver.language = "pfg"

    text = list(weaver.weave())
    assert text == [
        ';',
        '; Michael A.G. Aïvázis',
        '; Orthologue',
        '; (c) 1998-2019 All Rights Reserved',
        ';',
        '',
        '',
        '; end of file'
        ]

    return


# main
if __name__ == "__main__":
    test()


# end of file
