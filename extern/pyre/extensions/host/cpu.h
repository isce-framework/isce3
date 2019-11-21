// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_host_cpu_h)
#define pyre_extensions_host_cpu_h

// create my namespace
namespace pyre {
    namespace extensions {
        namespace host {
            // declarations
            const char * const logical__name__ = "logical";
            const char * const logical__doc__ = "the number of logical processors";
            PyObject * logical(PyObject *, PyObject *);

            const char * const logicalMax__name__ = "logicalMax";
            const char * const logicalMax__doc__ = "the maximum number of logical processors";
            PyObject * logicalMax(PyObject *, PyObject *);

            const char * const physical__name__ = "physical";
            const char * const physical__doc__ = "the number of physical processors";
            PyObject * physical(PyObject *, PyObject *);

            const char * const physicalMax__name__ = "physicalMax";
            const char * const physicalMax__doc__ = "the maximum number of physical processors";
            PyObject * physicalMax(PyObject *, PyObject *);
        } // of namespace host
    } // of namespace extensions
} // of namespace pyre

# endif

// end of file
