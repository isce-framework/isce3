// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Inventory)
#define pyre_journal_Inventory

// place Inventory in namespace pyre::journal
namespace pyre {
    namespace journal {
        template <bool = true> class Inventory;
        class Device;
    }
}


// declaration
template <bool DefaultState>
class pyre::journal::Inventory {
    // types
public:
    using state_t = bool;
    using device_t = Device;

    // interface
public:
    // accessors
    inline state_t state() const;
    inline device_t * device() const;

    // mutators
    inline void activate();
    inline void deactivate();
    inline void device(device_t *);

    // meta methods
public:
    inline ~Inventory();
    inline Inventory(state_t = DefaultState, device_t * = 0);
    inline Inventory(const Inventory &);
    inline const Inventory & operator=(const Inventory &);

    // data members
private:
    state_t _state;
    device_t * _device;
};


// get the inline definitions
#define pyre_journal_Inventory_icc
#include "Inventory.icc"
#undef pyre_journal_Inventory_icc


# endif
// end of file
