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

        // compute the size of the sliced group
        int size = sliced.size();
        // and my rank
        int rank = sliced.rank();

        // check
        // the new group size if half the original
        assert(size == (wsize+1)/2);
        // check my rank in the new group
        if (wrank % 2 == 0) {
            assert (rank == wrank / 2);
        } else {
            assert(rank == MPI_UNDEFINED);
        }
    }

    // shutdown
    MPI_Finalize();

    // all done
    return 0;
}

// end of file
