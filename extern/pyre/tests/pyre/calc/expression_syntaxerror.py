#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that syntax errors in expressions are caught
"""


def test():
    import pyre.calc

    # build a model
    model = pyre.calc.model()

    # unbalanced open brace
    try:
        model.expression(value="{production", model=model)
        assert False
    except model.ExpressionSyntaxError:
        pass

    # unbalanced open brace
    try:
        model.expression(value="production}", model=model)
        assert False
    except model.ExpressionSyntaxError:
        pass

    # unbalanced parenthesis
    try:
        model.expression(value="{production}({shipping}", model=model)
        assert False
    except model.ExpressionSyntaxError:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
