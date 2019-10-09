#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise component resolution
"""


def test():
    # access the framework
    import pyre
    # get the fileserver
    fs = pyre.executive.fileserver

    # resolve a trivial component
    c = pyre.resolve(uri='import:pyre.component')
    # check it
    assert c is pyre.component

    # something a bit more difficult
    joe = pyre.resolve(uri='file:sample.py/worker#joe')
    # check it
    assert isinstance(joe, pyre.component)
    assert joe.pyre_name == 'joe'

    # through the vfs
    barry = pyre.resolve(uri='vfs:{}/sample.py/worker#barry'.format(fs.STARTUP_DIR))
    # check it
    assert isinstance(barry, pyre.component)
    assert barry.pyre_name == 'barry'

    # using {import} instead of {file}
    bob = pyre.resolve(uri='import:sample.worker#bob')
    # check it
    assert isinstance(bob, pyre.component)
    assert bob.pyre_name == 'bob'

    # using {import} implicitly
    fred = pyre.resolve(uri='sample.worker#fred')
    # check it
    assert isinstance(bob, pyre.component)
    assert fred.pyre_name == 'fred'

    # this should fail quietly
    p = pyre.resolve(uri='import:pyre.protocol')
    # check it
    assert p is None

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
