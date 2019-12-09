#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the path primitive
"""


def test():
    # the home of the factory
    import pyre.primitives

    # the location of this test
    cwd = pyre.primitives.path.cwd()
    # the location with the crazy links that {mm} prepared
    scratch = cwd / 'scratch'

    # a couple of easy ones
    # this is guaranteed by posix
    # assert cwd == cwd.resolve()
    # this is an absolute path with no links
    # assert scratch == scratch.resolve()

    # another good link
    here = scratch / 'here'
    # check
    # assert scratch == here.resolve()

    # another good link
    up = scratch / 'up'
    # check
    # assert cwd == up.resolve()

    # a cycle
    cycle = scratch / 'cycle'
    # check that
    try:
        # this fails
        print(cycle.resolve())
        # so we can't reach here
        assert False
    # further, check that
    except cycle.SymbolicLinkLoopError as error:
        # the message we expect
        msg = "while resolving '{0.path}': symbolic link loop at '{0.loop}'".format(error)
        # is what we get
        assert str(error) == msg

    # a loop
    loop = scratch / 'loop'
    # check that
    try:
        # this fails
        loop.resolve()
        # so we can't reach here
        assert False
    # further, check that
    except loop.SymbolicLinkLoopError as error:
        # the message we expect
        msg = "while resolving '{0.path}': symbolic link loop at '{0.loop}'".format(error)
        # is what we get
        assert str(error) == msg

    # a ramp
    ramp = scratch / 'ramp'
    # check that
    try:
        # this fails
        ramp.resolve()
        # so we can't reach here
        assert False
    # further, check that
    except ramp.SymbolicLinkLoopError as error:
        # the message we expect
        msg = "while resolving '{0.path}': symbolic link loop at '{0.loop}'".format(error)
        # is what we get
        assert str(error) == msg

    # a two link cycle
    tic = scratch / 'tic'
    # check that
    try:
        # this fails
        tic.resolve()
        # so we can't reach here
        assert False
    # further, check that
    except tic.SymbolicLinkLoopError as error:
        # the message we expect
        msg = "while resolving '{0.path}': symbolic link loop at '{0.loop}'".format(error)
        # is what we get
        assert str(error) == msg

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
