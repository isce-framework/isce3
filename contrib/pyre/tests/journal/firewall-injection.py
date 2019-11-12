#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that {firewall.log} works as expected
"""


def test():
    # access the package
    import journal
    # build a firewall channel
    firewall = journal.firewall("journal.test1")
    # deactivate it
    firewall.active = False

    # and make it say something
    firewall.log("hello world!")

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
