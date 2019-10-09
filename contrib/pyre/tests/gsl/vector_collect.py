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

    # build my contribution
    θ = gsl.vector(shape=workload)
    # and initialize it
    for index in range(samplesPerTask):
        θ[index] = rank*samplesPerTask + index

    # decide on the destination task
    destination = 0
    # exercise it
    result = gsl.vector.collect(vector=θ, communicator=world, destination=destination)

    # at the destination task
    if rank == destination:
        # verify that i got the correct parts
        for task in range(tasks):
            for index in range(samplesPerTask):
                offset = task*samplesPerTask+index
                assert result[offset] == offset
        # print it out
        # result.print(format='{}')

    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
