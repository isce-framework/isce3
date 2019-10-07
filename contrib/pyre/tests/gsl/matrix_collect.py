#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the collector
"""


def test():
    # setup the workload
    sampleSize = 4
    samplesPerTask = 1
    workload = (samplesPerTask, sampleSize)

    # externals
    import mpi
    import gsl

    # get the world communicator
    world = mpi.world
    # figure out its geometry
    rank = world.rank
    tasks = world.size

    # build my contribution
    θ = gsl.matrix(shape=workload)
    # and initialize it
    for row in range(samplesPerTask):
        for column in range(sampleSize):
            θ[row, column] = (rank*samplesPerTask+row)*sampleSize + column

    # decide on the destination task
    destination = 0
    # exercise it
    result = gsl.matrix.collect(matrix=θ, communicator=world, destination=destination)

    # at the destination task
    if rank == destination:
        # verify that i got the correct parts
        for task in range(tasks):
            for sample in range(samplesPerTask):
                for dof in range(sampleSize):
                    offset = task*samplesPerTask+sample
                    assert result[offset, dof] == offset*sampleSize + dof
        # print it out
        # result.print(format='{}')

    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
