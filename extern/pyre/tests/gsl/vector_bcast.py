#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the partitioner
"""


def test():
    # setup the workload
    parameters = 8

    # externals
    import mpi
    import gsl

    # get the world communicator
    world = mpi.world
    # figure out its geometry
    rank = world.rank
    tasks = world.size

    # decide which task is the source
    source = 0
    # at the source task
    if rank == source:
        # allocate a matrix
        θ = gsl.vector(shape=parameters)
        # initialize it
        for dof in range(parameters):
            θ[dof] = dof
        # print it out
        # θ.print(format="{}")
    # the other tasks
    else:
        # have a dummy source matrix
        θ = None

    # broadcast
    result = gsl.vector.bcast(source=source, vector=θ)

    # verify that i got the correct part
    for dof in range(parameters):
        assert result[dof] == dof

    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
