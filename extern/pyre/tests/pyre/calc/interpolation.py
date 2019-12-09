#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that expressions work
"""


def test():
    import os
    import pyre.calc

    # set up the model
    model = pyre.calc.model()

    home = '/opt/local'
    model["tools"] = home
    model["bin"] = model.interpolation("{tools}/bin")
    model["lib"] = model.interpolation("{tools}/lib")
    model["include"] = model.interpolation("{tools}/include")

    # check the values
    # print("before:")
    # print("  tools: {!r}".format(model["tools"]))
    # print("  bin: {!r}".format(model["bin"]))
    # print("  lib: {!r}".format(model["lib"]))
    # print("  include: {!r}".format(model["include"]))
    assert model["tools"] == home
    assert model["bin"] == os.path.join(home, 'bin')
    assert model["lib"] == os.path.join(home, 'lib')
    assert model["include"] == os.path.join(home, 'include')

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
