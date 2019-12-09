#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify nodes with aggregator evaluators
"""


def test():
    import pyre.calc

    # make some nodes
    nodes = []
    for n in range(10):
        node = pyre.calc.var(value=n)
        nodes.append(node)

    count = pyre.calc.count(*nodes)
    sum = pyre.calc.sum(*nodes)
    min = pyre.calc.min(*nodes)
    max = pyre.calc.max(*nodes)
    average = pyre.calc.average(*nodes)

    # check
    assert count.value == 10
    assert sum.value == 45
    assert min.value == 0
    assert max.value == 9
    assert average.value == 4.5

    return


# main
if __name__ == "__main__":
    # request debugging support for the pyre.calc package
    pyre_debug = { "pyre.calc" }
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()
    # verify reference counts
    # for nodes
    from pyre.calc.Node import Node
    # print(tuple(Node._pyre_extent))
    assert tuple(Node._pyre_extent) == ()


# end of file
