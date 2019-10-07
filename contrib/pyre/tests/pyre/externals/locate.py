#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Ask the manager for a package based only on its category
"""


def test():
    # externals
    import os
    # get the framework
    import pyre
    # get the registered package manager
    manager = pyre.executive.host.packager
    # get the python package manage
    from pyre.externals.Python import Python

    # look for python; it's built-in so at least one instance must be there
    for python in manager.packages(category=Python):
        # check that we got something
        assert python
        # check that we pulled packages from the right category
        assert python.category == 'python'
        # show me where it's from
        print('python: {.pyre_spec}'.format(python))
        print('  binaries: {}'.format(python.join(python.bindir)))
        print('  headers: {}'.format(python.join(python.incdir)))
        print('  libraries: {}'.format(python.join(python.libdir)))
        print('  interpreter: {.interpreter}'.format(python))
        # look no further
        break

    # all done
    return manager


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
