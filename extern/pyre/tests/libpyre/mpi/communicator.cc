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
        // and my rank
        int wrank = whole.rank();

        // select a subset of the ranks, say the even ones
        std::vector<int> ranks;
        for (int rank = 0; rank < wsize; ++rank) {
            if (rank % 2 == 0) {
                ranks.push_back(rank);
            }
        }

        // build the new group
        group_t sliced = whole.include(ranks);
        // and a new communicator
        communicator_t newcom = world.communicator(sliced);

        // if my world rank is even
        if (wrank % 2 == 0) {
            // compute the size of the new communicator
            int size = newcom.size();
            // and my rank in it
            int rank = newcom.rank();
            // check
            // the new communicator size if half the original
            assert(size == wsize/2);
            // so is my rank
            assert (rank == wrank / 2);
        } else {
            // otherwise, verify that i got a null communicator
            assert(newcom.isNull());
        }
    }

    // shutdown
    MPI_Finalize();

    // all done
    return 0;
}

// end of file
