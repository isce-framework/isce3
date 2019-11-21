#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise an SVG weaver
"""


def test():
    # get the package
    import pyre.weaver
    # instantiate a weaver
    weaver = pyre.weaver.weaver(name="sanity")
    weaver.language = "svg"

    text = list(weaver.weave())
    assert text == [
        '<?xml version="1.0"?>',
        '<!--',
        ' ! ',
        ' ! Michael A.G. Aïvázis',
        ' ! Orthologue',
        ' ! (c) 1998-2019 All Rights Reserved',
        ' ! ',
        ' -->',
        '',
        '<svg version="1.1" xmlns="http://www.w3.org/2000/svg">',
        '',
        '',
        '</svg>',
        '',
        '<!-- end of file -->'
        ]

    return


# main
if __name__ == "__main__":
    test()


# end of file
