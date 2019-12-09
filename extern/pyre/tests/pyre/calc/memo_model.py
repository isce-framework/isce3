#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise automatic updates for interpolations
"""

# externals
import os


def test():
    # access the package
    import pyre.calc
    # set up the model
    model = pyre.calc.model()

    # build some nodes
    home = '/opt/local'
    model["tools"] = home
    model["bin"] = bin = model.interpolation("{tools}/bin")
    model["lib"] = lib = model.interpolation("{tools}/lib")
    model["include"] = include = model.interpolation("{tools}/include")

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

    # make a change
    home = '/usr'
    model["tools"] = home

    # check that the dependents are now dirty
    assert bin.dirty == True
    assert lib.dirty == True
    assert include.dirty == True

    # check the values
    # print("after:")
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
    # do...
    test()


# end of file
