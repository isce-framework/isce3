#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that {info.log} works as expected
"""


def test():
    # access the package
    import journal
    # build a info channel
    info = journal.info("journal.test1")
    # deactivate it
    info.active = False

    # and make it say something
    info.log("hello world!")

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
