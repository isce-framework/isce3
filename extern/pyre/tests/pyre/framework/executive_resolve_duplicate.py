#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the binder can retrieve components from odb files
"""


def test():
    import pyre
    executive =  pyre.executive

    # retrieve a component descriptor from a file
    one, *_ = executive.resolve(uri="file:sample.odb/one")
    two, *_ = executive.resolve(uri="file:sample.odb/one")
    # check that the two retrievals yield identical results
    assert one == two

    # all done
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
