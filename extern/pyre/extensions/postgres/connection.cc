// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <iostream>

#include <Python.h>
#include <libpq-fe.h>
#include <pyre/journal.h>

#include "connection.h"
#include "constants.h"
#include "interlayer.h"


namespace pyre {
    namespace extensions {
        namespace postgres {
            // helpers
            void finish(PyObject *);

            // constants
            const char * const connectionCapsuleName = "postgres.connection";
        } // of namespace postgres
    } // of namespace extensions
} // of namespace pyre


// establish a new connection
const char * const
pyre::extensions::postgres::
connect__name__ = "connect";

const char * const
pyre::extensions::postgres::
connect__doc__ = "establish a connection to the postgres back end";

PyObject *
pyre::extensions::postgres::
connect(PyObject *, PyObject * args) {
    // the connection specification
    const char * specification;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:connect", &specification)) {
        return 0;
    }

    // in case someone is listening...
    pyre::journal::debug_t debug("postgres.connection");
    debug
        << pyre::journal::at(__HERE__)
        << "connecting with specification: '" << specification << "'"
        << pyre::journal::endl;

    // establish a connection
    PGconn * connection = PQconnectdb(specification);
    // check
    if (PQstatus(connection) != CONNECTION_OK) {
        // convert the error to human readable form
        const char * description = PQerrorMessage(connection);
        // according to DB API 2.0, connection errors are OperationalError
        return raiseOperationalError(description);
    }

    return PyCapsule_New(connection, connectionCapsuleName, finish);
}


const char * const
pyre::extensions::postgres::
disconnect__name__ = "disconnect";

const char * const
pyre::extensions::postgres::
disconnect__doc__ = "shut down a connection to the postgres back end";

PyObject *
pyre::extensions::postgres::
disconnect(PyObject *, PyObject * args) {
    // the connection capsule
    PyObject * connection;
    // extract it from the arguments
    if (!PyArg_ParseTuple(args, "O!:disconnect", &PyCapsule_Type, &connection)) {
        return 0;
    }
    // call the destructor
    finish(connection);
    // and remove the destructor
    PyCapsule_SetDestructor(connection, 0);

    // in case someone is listening...
    pyre::journal::debug_t debug("postgres.connection");
    debug
        << pyre::journal::at(__HERE__)
        << "disconnected"
        << pyre::journal::endl;

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}


// shutdown an existing connection
void
pyre::extensions::postgres::
finish(PyObject * capsule) {
    // bail if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, connectionCapsuleName)) {
        return;
    }
    // get pointer from the capsule and cast it to a pg connection
    PGconn * connection =
        static_cast<PGconn *>(PyCapsule_GetPointer(capsule, connectionCapsuleName));

    // in case someone is listening...
    pyre::journal::debug_t debug("postgres.connection");
    debug
        << pyre::journal::at(__HERE__)
        << "closing the connection to the server"
        << pyre::journal::endl;

    // shutdown
    PQfinish(connection);
    // all done
    return;
}


// end of file
