// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_mpi_Communicator_h)
#define pyre_mpi_Communicator_h

// place Communicator in namespace pyre::mpi
namespace pyre {
    namespace mpi {
        class Communicator;
        class Error;
        class Group;
    }
}

// declaration
class pyre::mpi::Communicator {
    friend class Shareable<Communicator>;

    // types
public:
    typedef MPI_Comm handle_t;
    typedef Handle<Communicator> storage_t;
    typedef Shareable<Communicator> shared_t;

    typedef Group group_t;
    typedef Communicator communicator_t;
    typedef std::vector<int> ranklist_t;

    // interface
public:
    inline handle_t handle() const;
    inline bool isNull() const;

    inline void barrier() const; // build a synchronization barrier

    inline int rank() const; // compute the rank of this process
    inline int size() const; // compute my size

    inline group_t group() const; // access to my group of processes
    inline communicator_t communicator(const group_t &) const;

    inline communicator_t cartesian(const ranklist_t &, const ranklist_t &, int) const;
    inline ranklist_t coordinates(int) const;

    // meta methods
public:
    inline ~Communicator();
    inline Communicator(handle_t, bool = false);
    inline Communicator(const Communicator &);
    inline const Communicator & operator=(const Communicator &);

    // hidden
private:
    static inline void free(MPI_Comm *);

    // data members
private:
    storage_t _handle;
};


// get the inline definitions
#define pyre_mpi_Communicator_icc
#include "Communicator.icc"
#undef pyre_mpi_Communicator_icc


# endif
// end of file
