#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise python's understanding of the command line
"""


def test():
    import sys
    for index, arg in enumerate(sys.argv):
        print(index, arg, sep=": ")
    return


# main
if __name__ == "__main__":
    test()


# end of file
