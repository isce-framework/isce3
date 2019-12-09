// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_mpi_Group_h)
#define pyre_mpi_Group_h

// place Group in namespace pyre::mpi
namespace pyre {
    namespace mpi {
        class Group;
        class Communicator;
        class Error;

        inline Group groupUnion(const Group &, const Group &);
        inline Group groupIntersection(const Group &, const Group &);
        inline Group groupDifference(const Group &, const Group &);
    }
}

// declaration
class pyre::mpi::Group {
    friend class Communicator;
    friend class Shareable<Group>;

    friend Group groupUnion(const Group &, const Group &);
    friend Group groupIntersection(const Group &, const Group &);
    friend Group groupDifference(const Group &, const Group &);

    // types
public:
    typedef MPI_Group handle_t;
    typedef Handle<Group> storage_t;
    typedef Shareable<Group> shared_t;

    typedef Group group_t;
    typedef std::vector<int> ranklist_t;

    // interface
public:
    inline bool isEmpty() const;
    inline int rank() const;
    inline int size() const;

    inline group_t include(const ranklist_t &) const;
    inline group_t exclude(const ranklist_t &) const;

    // meta methods
public:
    inline ~Group();
    inline Group(handle_t handle, bool = false);
    inline Group(const Group &);
    inline const Group & operator=(const Group &);

    // hidden
private:
    inline operator handle_t () const;
    static inline void free(MPI_Group *);

    // data members
private:
    storage_t _handle;
};


// get the inline definitions
#define pyre_mpi_Group_icc
#include "Group.icc"
#undef pyre_mpi_Group_icc


# endif
// end of file
