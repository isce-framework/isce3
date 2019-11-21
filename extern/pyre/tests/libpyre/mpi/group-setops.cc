// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>
//  other packages
#include <cassert>
// grab the mpi objects
#include <pyre/mpi.h>

typedef pyre::mpi::group_t group_t;
typedef pyre::mpi::communicator_t communicator_t;

// main program
int main() {
    // initialize
    MPI_Init(0, 0);

    // push down a scope to make sure our local variables get destroyed before MPI_Finalize
    {
        // build a handle to the world communicator
        communicator_t world(MPI_COMM_WORLD, true);
        // get its group
        group_t whole = world.group();
        // compute the size of the world group
        int wsize = whole.size();

        // select a subset of the ranks, say the even ones
        std::vector<int> evens;
        for (int rank = 0; rank < wsize; ++rank) {
            if (rank % 2 == 0) {
                evens.push_back(rank);
            }
        }

        // build the new groups
        group_t even = whole.include(evens);
        group_t odd = whole.exclude(evens);

        // compute the union of these two
        group_t gu = pyre::mpi::groupUnion(even, odd);
        // verify that it is the right size
        assert(gu.size() == world.size());

        // compute the intersection
        group_t gi = pyre::mpi::groupIntersection(even, odd);
        // verify it is empty
        assert(gi.isEmpty());

        // compute the difference world - odd
        group_t gd = pyre::mpi::groupDifference(whole, odd);
        // verify it is the same size as the even group
        assert(gd.size() == even.size());

    }

    // shutdown
    MPI_Finalize();

    // all done
    return 0;
}

// end of file
