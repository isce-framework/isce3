// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_mpi_h)
#define pyre_mpi_h


// external packages
#include <mpi.h>
#include <vector>
#include <exception>

// local types
#include "mpi/Error.h"

#include "mpi/Shareable.h"
#include "mpi/Handle.h"

#include "mpi/Group.h"
#include "mpi/Communicator.h"


// type declarations
namespace pyre {
    namespace mpi {
        typedef Error error_t;
        typedef Communicator communicator_t;
        typedef Group group_t;
    }
}

#endif


// end of file
