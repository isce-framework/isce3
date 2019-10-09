#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


"""
Verify that modifying a list while iterating over it is ok
"""


def test():
    done = []
    todo = [0]

    for n in todo:
        done.append(n)
        if n < 10:
            todo.append(n+1)

    assert done == list(range(11))

    return


# main
if __name__ == "__main__":
    test()


# end of file
