// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <map>
#include <gsl/gsl_rng.h>

#include "rng.h"
#include "capsules.h"

#include <iostream>

// the table of generators
namespace gsl {
    namespace rng {
        typedef std::map<std::string, const gsl_rng_type *> map_t;
        static map_t generators;
    }
}


// get the name of all the generators known to GSL
const char * const gsl::rng::avail__name__ = "rng_avail";
const char * const gsl::rng::avail__doc__ = "return the set of all known generators";

PyObject *
gsl::rng::avail(PyObject *, PyObject * args) {
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, ":rng_avail");
    // if something went wrong
    if (!status) return 0;

    // make a frozen set to hold the names
    PyObject *names = PyFrozenSet_New(0);
    // iterate over the registered names
    for (
         gsl::rng::map_t::const_iterator i= gsl::rng::generators.begin();
         i != gsl::rng::generators.end();
         i++ ) {
        // add the name to the set
        PySet_Add(names, PyUnicode_FromString(i->first.c_str()));
    }

    // return the names
    return names;
}


// construction
const char * const gsl::rng::alloc__name__ = "rng_alloc";
const char * const gsl::rng::alloc__doc__ = "allocate a rng";

PyObject *
gsl::rng::alloc(PyObject *, PyObject * args) {
    // place holders for the python arguments
    char * name;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "s:rng_alloc", &name);
    // if something went wrong
    if (!status) return 0;

    // get the rng type
    const gsl_rng_type *algorithm = gsl::rng::generators[name];
    // std::cout << "{" << name << "} -> " << algorithm << std::endl;
    // if it's not in table
    if (!algorithm) {
        PyErr_SetString(PyExc_ValueError, "unknown random number generator");
        return 0;
    }

    // allocate a rng
    gsl_rng * r = gsl_rng_alloc(algorithm);
    // std::cout << " gsl.rng_allocate: rng@" << r << ", name=" << name << std::endl;

    // wrap it in a capsule and return it
    return PyCapsule_New(r, capsule_t, free);
}


// seeding
const char * const gsl::rng::set__name__ = "rng_set";
const char * const gsl::rng::set__doc__ = "seed a random number generator";

PyObject *
gsl::rng::set(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    unsigned long seed;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!k:rng_set", &PyCapsule_Type, &capsule, &seed);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid rng capsule");
        return 0;
    }

    // get the rng
    gsl_rng * r = static_cast<gsl_rng *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.rng_set: rng@" << r << std::endl;
    // seed it
    gsl_rng_set(r, seed);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// get the name of the generator as known to GSL
const char * const gsl::rng::name__name__ = "rng_name";
const char * const gsl::rng::name__doc__ = "look up the name of the generator";

PyObject *
gsl::rng::name(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:rng_name", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid rng capsule");
        return 0;
    }

    // get the rng
    gsl_rng * r = static_cast<gsl_rng *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.rng_name: rng@" << r << std::endl;

    // get the name
    const char * name = gsl_rng_name(r);

    // return the name
    return PyUnicode_FromString(name);
}


// get the range of generated values
const char * const gsl::rng::range__name__ = "rng_range";
const char * const gsl::rng::range__doc__ =
    "return a tuple (min, max) describing the range of values generated";

PyObject *
gsl::rng::range(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:rng_name", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid rng capsule");
        return 0;
    }

    // get the rng
    gsl_rng * r = static_cast<gsl_rng *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.rng_range: rng@" << r << std::endl;

    unsigned long min;
    unsigned long max;

    // compute
    min = gsl_rng_min(r);
    max = gsl_rng_max(r);

    // build the range
    PyObject * range = PyTuple_New(2);
    PyTuple_SET_ITEM(range, 0, PyLong_FromUnsignedLong(min));
    PyTuple_SET_ITEM(range, 1, PyLong_FromUnsignedLong(max));

    // return the range
    return range;
}


// get the next random integer
const char * const gsl::rng::get__name__ = "rng_get";
const char * const gsl::rng::get__doc__ =
    "return the next random integer with the range of the generator";

PyObject *
gsl::rng::get(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:rng_get", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid rng capsule");
        return 0;
    }

    // get the rng
    gsl_rng * r = static_cast<gsl_rng *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.rng_range: rng@" << r << std::endl;

    // get
    unsigned long v = gsl_rng_get(r);

    // return a value
    return PyLong_FromUnsignedLong(v);
}


// a random double in [0,1)
const char * const gsl::rng::uniform__name__ = "rng_uniform";
const char * const gsl::rng::uniform__doc__ =
    "return the next random integer with the range of the generator";

PyObject *
gsl::rng::uniform(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:rng_uniform", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid rng capsule");
        return 0;
    }

    // get the rng
    gsl_rng * r = static_cast<gsl_rng *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.rng_range: rng@" << r << std::endl;

    // get a value
    double value = gsl_rng_uniform(r);

    // return a value
    return PyFloat_FromDouble(value);
}


// helpers
// initialization of the known generators
void
gsl::rng::initialize()
{
    // std::cout << " ++ gsl.rng.initialize:" << std::endl;
    // loop over all the  registered types
    for (const gsl_rng_type **current=gsl_rng_types_setup(); (*current)!=0; current++) {
        // add each one to my map
        // std::cout << "      {" << (*current)->name << "}" << std::endl;
        gsl::rng::generators[(*current)->name] = *current;
    }

    // brag
    // std::cout
        // << " -- initialized " << gsl::rng::generators.size() << " generators"
        // << std::endl;

    return;
}


// destructor
void
gsl::rng::free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::rng::capsule_t)) return;
    // get the rng
    gsl_rng * r = static_cast<gsl_rng *>(PyCapsule_GetPointer(capsule, gsl::rng::capsule_t));
    // std::cout << " gsl.rng_free: rng@" << r << std::endl;
    // deallocate
    gsl_rng_free(r);
    // and return
    return;
}


// end of file
