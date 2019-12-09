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
    sampleSize = 4
    samplesPerTask = 4
    workload = (samplesPerTask, sampleSize)

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
        θ = gsl.matrix(shape=(tasks*samplesPerTask, sampleSize))
        # initialize it
        for task in range(tasks):
            for sample in range(samplesPerTask):
                for dof in range(sampleSize):
                    offset = task*samplesPerTask + sample
                    θ[offset, dof] = offset*sampleSize + dof
        # print it out
        # θ.print(format="{}")
    # the other tasks
    else:
        # have a dummy source matrix
        θ = None

    # build the destination matrix
    part = gsl.matrix(shape=workload)
    # make a partition
    part.excerpt(communicator=world, source=source, matrix=θ)

    # verify that i got the correct part
    for row in range(samplesPerTask):
        for column in range(sampleSize):
            assert part[row, column] == (rank*samplesPerTask+row)*sampleSize + column

    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
