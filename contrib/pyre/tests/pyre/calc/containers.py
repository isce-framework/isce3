#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify container nodes
"""


def test():
    import pyre.calc

    # make some nodes
    nodes = []
    for n in range(10):
        node = pyre.calc.var(value=n+1)
        nodes.append(node)

    # make a sequence
    s = pyre.calc.sequence(*nodes)
    # check
    assert tuple(s.value) == tuple(range(1, 11))

    # make a mapping
    m = pyre.calc.mapping(**{ 'n{}'.format(i): n for i,n in enumerate(nodes)})
    # check
    assert m.value == dict({ 'n{}'.format(i): i+1 for i in range(10)})

    # all done
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
