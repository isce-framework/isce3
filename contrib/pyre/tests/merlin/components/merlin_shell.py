#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the merlin shell is accessible
"""


def test():
    # package access
    import os
    from merlin import merlin

    # debug:
    # merlin.vfs.dump()
    # merlin.nameserver.dump('merlin')

    # retrieve project directory
    project = merlin.vfs['/project']
    # check it
    assert str(project.uri) == os.getcwd()

    # retrieve the location of the merlin settings
    cfg = merlin.vfs['/merlin/project']
    # check it
    assert str(cfg.uri) == os.path.join(os.getcwd(), '.merlin')
    # verify that it is accessible through the {pfs]
    assert cfg is merlin.pfs['/project']

    # and return
    return


# main
if __name__ == "__main__":
    test()


# end of file
