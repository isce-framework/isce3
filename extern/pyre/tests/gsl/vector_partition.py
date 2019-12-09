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
    samplesPerTask = 8
    workload = samplesPerTask

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
        # allocate a vector
        θ = gsl.vector(shape=tasks*samplesPerTask)
        # initialize it
        for task in range(tasks):
            for sample in range(samplesPerTask):
                offset = task*samplesPerTask + sample
                θ[offset] = offset
        # print it out
        # θ.print(format="{}")
    # the other tasks
    else:
        # have a dummy source vector
        θ = None

    # make a partition
    part = gsl.vector(shape=workload)
    part.excerpt(communicator=world, source=source, vector=θ)

    # verify that i got the correct part
    for index in range(samplesPerTask):
        assert part[index] == rank*samplesPerTask + index

    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
