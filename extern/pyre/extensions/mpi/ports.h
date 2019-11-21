// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_mpi_ports_h)
#define pyre_extensions_mpi_ports_h

// place everything in my private namespace
namespace mpi {
    namespace port {

        // send bytes
        extern const char * const sendBytes__name__;
        extern const char * const sendBytes__doc__;
        PyObject * sendBytes(PyObject *, PyObject *);

        // receive bytes
        extern const char * const recvBytes__name__;
        extern const char * const recvBytes__doc__;
        PyObject * recvBytes(PyObject *, PyObject *);

        // send a string
        extern const char * const sendString__name__;
        extern const char * const sendString__doc__;
        PyObject * sendString(PyObject *, PyObject *);

        // receive a string
        extern const char * const recvString__name__;
        extern const char * const recvString__doc__;
        PyObject * recvString(PyObject *, PyObject *);

    } // of namespace port
} // of namespace mpi

#endif

// end of file
