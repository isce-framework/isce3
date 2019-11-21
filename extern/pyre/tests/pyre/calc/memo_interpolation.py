#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise expression formula updates
"""

# externals
import os


def test():
    # access the package
    import pyre.calc
    # set up the model
    model = pyre.calc.model()

    # make some nodes
    home = model.variable(value='/opt/local')
    tools = model.interpolation("{home}")
    bin = model.interpolation("{tools}/bin")
    lib = model.interpolation("{tools}/lib")

    # add them to the model
    model["home"] = home
    model["tools"] = tools
    model["bin"] = bin
    model["lib"] = lib

    # check that they are all dirty
    assert tools.dirty == True
    assert bin.dirty == True
    assert lib.dirty == True
    # check the values
    assert home.value == '/opt/local'
    assert tools.value == home.value
    assert bin.value == os.path.join(tools.value, 'bin')
    assert lib.value == os.path.join(tools.value, 'lib')
    # check that they are all clean
    assert tools.dirty == False
    assert bin.dirty == False
    assert lib.dirty == False

    # make a formula change
    tools.value = "{home}/private"
    # check that they are all dirty
    assert tools.dirty == True
    assert bin.dirty == True
    assert lib.dirty == True
    # check the values
    assert tools.value == os.path.join(home.value, 'private')
    assert bin.value == os.path.join(tools.value, 'bin')
    assert lib.value == os.path.join(tools.value, 'lib')
    # check that they are all clean
    assert tools.dirty == False
    assert bin.dirty == False
    assert lib.dirty == False

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
