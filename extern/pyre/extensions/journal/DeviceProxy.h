// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_extensions_journal_DeviceProxy_h)
#define pyre_extensions_journal_DeviceProxy_h

// place DeviceProxy in namespace pyre::journal
namespace pyre {
    namespace extensions {
        namespace journal {
            class DeviceProxy;
        }
    }
}

// declaration
class pyre::extensions::journal::DeviceProxy : public pyre::journal::Device {
    // interface
public:
    virtual void record(entry_t &, metadata_t &);

    // meta methods
public:
    virtual ~DeviceProxy();
    inline DeviceProxy(PyObject * keeper);
    // disallow
private:
    DeviceProxy(const DeviceProxy &);
    const DeviceProxy & operator=(const DeviceProxy &);

    // data
private:
    PyObject * _keeper;
};


// get the inline definitions
#define pyre_extensions_journal_DeviceProxy_icc
#include "DeviceProxy.icc"
#undef pyre_extensions_journal_DeviceProxy_icc

# endif

// end of file
