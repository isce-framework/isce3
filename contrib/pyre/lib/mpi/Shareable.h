// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_mpi_Shareable_h)
#define pyre_mpi_Shareable_h

// place Shareable in namespace pyre::mpi
namespace pyre {
    namespace mpi {
        template <typename Resource> class Shareable;
    }
}


// declaration
template <typename Resource>
class pyre::mpi::Shareable {
    // types
public:
    typedef Resource resource_t;
    typedef typename Resource::handle_t handle_t;

    // interface
public:
    inline int acquire(); // increment the reference count
    inline int release(); // decrement the reference count
    inline int references() const; // return the number of outstanding references

    inline handle_t handle() const; // return the low level MPI handle

    // meta methods
public:
    inline ~Shareable();
    inline Shareable(handle_t, bool);

    // disallow the copy constructors
private:
    inline Shareable(const Shareable &);
    inline const Shareable & operator=(const Shareable &);

    // data members
private:
    int _count;
    bool _immortal;
    handle_t _handle;
};


// get the inline definitions
#define pyre_mpi_Shareable_icc
#include "Shareable.icc"
#undef pyre_mpi_Shareable_icc


# endif
// end of file
