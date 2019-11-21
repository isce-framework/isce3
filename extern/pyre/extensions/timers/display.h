// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_timers_display_h)
#define pyre_extensions_timers_display_h

// create my namespace
namespace pyre {
    namespace extensions {
        namespace timers {
            // declarations
            // timer factory
            const char * const newTimer__name__ = "newTimer";
            const char * const newTimer__doc__ = "construct a new timer";
            PyObject * newTimer(PyObject *, PyObject *);

            // start a timer
            const char * const start__name__ = "start";
            const char * const start__doc__ = "start a timer";
            PyObject * start(PyObject *, PyObject *);

            // stop a timer
            const char * const stop__name__ = "stop";
            const char * const stop__doc__ = "stop a timer";
            PyObject * stop(PyObject *, PyObject *);

            // reset a timer
            const char * const reset__name__ = "reset";
            const char * const reset__doc__ = "reset a timer";
            PyObject * reset(PyObject *, PyObject *);

            // take a reading from a running timer
            const char * const lap__name__ = "lap";
            const char * const lap__doc__ = "read a running timer";
            PyObject * lap(PyObject *, PyObject *);

            // take a reading from a stopped timer
            const char * const read__name__ = "read";
            const char * const read__doc__ = "read a stopped timer";
            PyObject * read(PyObject *, PyObject *);
        } // of namespace timers
    } // of namespace extensions
} // of namespace pyre

# endif

// end of file
