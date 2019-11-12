#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise an HTML weaver
"""


def test():
    # get the package
    import pyre.weaver
    # instantiate a weaver
    weaver = pyre.weaver.weaver(name="sanity")
    # with the right language
    weaver.language = "html"
    # render
    text = list(weaver.weave())
    # check
    assert text == [
        '<!doctype html>',
        '<!--',
        ' ! ',
        ' ! Michael A.G. Aïvázis',
        ' ! Orthologue',
        ' ! (c) 1998-2019 All Rights Reserved',
        ' ! ',
        ' -->',
        '<html>',
        '',
        '',
        '</html>',
        '<!-- end of file -->'
        ]
    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
