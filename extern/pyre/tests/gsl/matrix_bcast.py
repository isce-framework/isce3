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
    samples = 4
    parameters = 8
    workload = (samples, parameters)

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
        θ = gsl.matrix(shape=workload)
        # initialize it
        for sample in range(samples):
            for dof in range(parameters):
                θ[sample, dof] = sample*parameters + dof
        # print it out
        # θ.print(format="{}")
    # the other tasks
    else:
        # have a dummy source matrix
        θ = None

    # broadcast
    result = gsl.matrix.bcast(source=source, matrix=θ)

    # verify that i got the correct part
    for sample in range(samples):
        for dof in range(parameters):
            assert result[sample, dof] == sample*parameters + dof

    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
