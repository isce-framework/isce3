#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the device manager is accessible
"""


def test():
    # access the cuda module
    import cuda

    # show me
    # print(cuda.manager)
    # make sure there is a manager
    assert cuda.manager

    # show me
    # print("CUDA capable devices:")
    # for device in cuda.manager.devices:
        # device.dump(indent='  ')

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
