#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that syntax errors in interpolations are caught
"""


def test():
    import pyre.calc

    # build a model
    model = pyre.calc.model()

    # escaped macro delimiters
    node = model.interpolation('{{production}}')
    # verify it made a variable
    assert type(node) is model.node.variable

    # and another
    node = model.interpolation('{{{{cost per unit}}}}')
    # verify it made a variable
    assert type(node) is model.node.variable

    # finally
    tricky = model.interpolation('{{{number of items}}}')
    # verify it made a variable
    assert type(tricky) is model.node.interpolation
    # with an unresolved node
    try:
        tricky.value
        assert False
    except tricky.UnresolvedNodeError:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
