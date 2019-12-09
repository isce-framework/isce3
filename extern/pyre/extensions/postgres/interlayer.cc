// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>

#include <Python.h>
#include <libpq-fe.h>
#include <pyre/journal.h>

#include "exceptions.h"
#include "interlayer.h"
#include "constants.h"


// types
typedef PyObject * (*pythonizer_t)(const char *);

// declarations of the converters; definitions at the bottom
PyObject * asNone(const char *);
PyObject * asString(const char *);


// convert the tuples in PGresult into a python tuple
PyObject *
pyre::extensions::postgres::
buildResultTuple(PGresult * result)
{
    // set up the debug channel
    pyre::journal::debug_t debug("postgres.conversions");

    // find out how many tuples in the result
    int tuples = PQntuples(result);
    // and how many fields in each tuple
    int fields = PQnfields(result);
    // build a python tuple to hold the data
    PyObject * data = PyTuple_New(tuples+1);

    // build a tuple to hold the names of the fields
    PyObject * header = PyTuple_New(fields);
    // populate the header tuple with the names of the fields
    for (int field = 0; field < fields; field++) {
        // add the field name to the tuple
        PyTuple_SET_ITEM(header, field, PyUnicode_FromString(PQfname(result, field)));
    }
    // add the header to the data set
    PyTuple_SET_ITEM(data, 0, header);
    // bail out if the output is trivial
    if (tuples == 0) {
        return data;
    }

    // create a vector of data processors
    std::vector<pythonizer_t> converters;
    // iterate over the columns looking for the appropriate converter
    for (int field = 0; field < fields; field++) {
        // text, binary, or what?
        if (PQfformat(result, field) == 0) {
            debug
                << pyre::journal::at(__HERE__)
                << "field '" << PQfname(result, field) << "' is formatted as text"
                << pyre::journal::endl;
            converters.push_back(asString);

        } else if (PQfformat(result, field) == 1) {
            pyre::journal::firewall_t firewall("postgress.conversions");
            firewall
                << pyre::journal::at(__HERE__)
                << "NYI: binary data for field '"
                << PQfname(result, field)
                << "'"
                << pyre::journal::endl;
            converters.push_back(asNone);

        } else {
            pyre::journal::firewall_t firewall("postgress.conversions");
            firewall
                << pyre::journal::at(__HERE__)
                << "unknown postgres format code for field '"
                << PQfname(result, field)
                << "'"
                << pyre::journal::endl;
            converters.push_back(asNone);
        }
    }

    // iterate over the rows
    for (int tuple = 0; tuple < tuples; tuple++) {
        // build a tuple to hold this row
        PyObject * row = PyTuple_New(fields);
        // iterate over the data fields
        for (int field = 0; field < fields; field++) {

            // place-holder for the field value
            PyObject * item;
            // if it is not null
            if (!PQgetisnull(result, tuple, field)) {
                // extract it
                const char * value = PQgetvalue(result, tuple, field);
                item = converters[field](value);
            } else {
                // otherwise it is null, which we encode as None
                Py_INCREF(null);
                item = null;
            }
            // add it to the tuple
            PyTuple_SET_ITEM(row, field, item);
        }
        // and now that the row tuple is fully built, add it to the data set
        PyTuple_SET_ITEM(data, tuple+1, row);
    }

    // return the data tuple
    return data;
}


// analyze and process the result set
PyObject *
pyre::extensions::postgres::
processResult(string_t command, PGresult * result, resultProcessor_t processor)
{
    // in case someone is listening...
    pyre::journal::debug_t debug("postgres.execution");
    debug
        << pyre::journal::at(__HERE__)
        << "analyzing result set"
        << pyre::journal::endl;

    //  this is what we will return to the caller
    PyObject * value;
    // start looking
    if (!result) {
        // a null result signifies that there is nothing available from the server this can
        // happen when repeatedly calling {retrieve} to get the result of queries that contain
        // multiple SQL statements; it is not necessarily and error

        // return None
        Py_INCREF(Py_None);
        value = Py_None;

    } else if (PQresultStatus(result) == PGRES_COMMAND_OK) {
        // the command was executed successfully
        // diagnostics
        if (debug.isActive()) {
            debug
                << pyre::journal::at(__HERE__)
                << "success: " << PQcmdStatus(result)
                << pyre::journal::endl;
        }
        // build the return value
        Py_INCREF(Py_None);
        value = Py_None;

    } else if (PQresultStatus(result) == PGRES_TUPLES_OK) {
        // the query succeeded and there are tuples to harvest
        if (debug.isActive()) {
            int fields = PQnfields(result);
            int tuples = PQntuples(result);
            debug
                << pyre::journal::at(__HERE__)
                << "success: "
                << tuples << " tuple" << (tuples == 1 ? "" : "s")
                << " with " << fields << " field" << (fields == 1 ? "" : "s") << " each"
                << pyre::journal::endl;
        }
        // build the return value
        value = processor(result);
    } else {
        // there was something wrong with the command
        const char * description = PQresultErrorMessage(result);
        // raise a ProgrammingError
        value = raiseProgrammingError(description, command);
    }

    // all is well
    // free the result
    PQclear(result);
    // and return
    return value;
}


// support for raising exceptions
// raise an OperationalError exception
PyObject *
pyre::extensions::postgres::
raiseOperationalError(string_t description)
{
    PyObject * args = PyTuple_New(0);
    PyObject * kwds = Py_BuildValue("{s:s}", "description", description);
    PyObject * exception = PyObject_Call(OperationalError, args, kwds);
    // prepare to raise an instance of OperationalError
    PyErr_SetObject(OperationalError, exception);
    // and return an error indicator
    return 0;
}


// raise a ProgrammingError exception
PyObject *
pyre::extensions::postgres::
raiseProgrammingError(string_t description, string_t command)
{
    PyObject * args = PyTuple_New(0);
    PyObject * kwds = Py_BuildValue("{s:s, s:s}",
                                    "diagnostic", description,
                                    "command", command
                                    );
    PyObject * exception = PyObject_Call(ProgrammingError, args, kwds);
    // prepare to raise an instance of ProgrammingError
    PyErr_SetObject(ProgrammingError, exception);
    // and return an error indicator
    return 0;
}


// data converter definitions
 PyObject * asNone(const char * value)
 {
     Py_INCREF(Py_None);
     return Py_None;
 }

 PyObject * asString(const char * value)
 {
     return PyUnicode_FromString(value);
 }

// end of file
