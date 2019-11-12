#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Fork and exec the same script
"""


def test():
    import sys
    # if this is the forked child
    if "--child" in sys.argv:
        # spit out the command line
        # print("child:", sys.argv)
        # we are done
        return

    # otherwise, fork
    import os
    pid = os.fork()
    # in the parent process
    if pid > 0:
        # wait for the child to finish
        child, status = os.wait()
        # check that it was the right one
        assert pid == child
        # all done
        return
    # in the child process, build the new command line
    argv = [sys.executable] + sys.argv + ["--child"]
    # print("execv:", argv)
    # and exec
    return os.execv(sys.executable, argv)


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
