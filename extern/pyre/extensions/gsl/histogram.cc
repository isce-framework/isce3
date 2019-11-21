// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_histogram.h>

#include "histogram.h"
#include "capsules.h"


// construction
const char * const gsl::histogram::alloc__name__ = "histogram_alloc";
const char * const gsl::histogram::alloc__doc__ = "allocate a histogram";

PyObject *
gsl::histogram::alloc(PyObject *, PyObject * args) {
    // place holders for the python arguments
    size_t shape;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "k:histogram_alloc", &shape);
    // if something went wrong
    if (!status) return 0;

    // allocate a histogram
    gsl_histogram * h = gsl_histogram_alloc(shape);

    // wrap it in a capsule and return it
    return PyCapsule_New(h, capsule_t, free);
}


// initialization
const char * const gsl::histogram::uniform__name__ = "histogram_uniform";
const char * const gsl::histogram::uniform__doc__ =
    "build bins with uniform coverage of a given range";

PyObject *
gsl::histogram::uniform(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    double lower, upper;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!dd:histogram_uniform",
                                  &PyCapsule_Type, &capsule,
                                  &lower, &upper
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));
    // uniform it out
    gsl_histogram_set_ranges_uniform(h, lower, upper);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// ranges
const char * const gsl::histogram::ranges__name__ = "histogram_ranges";
const char * const gsl::histogram::ranges__doc__ =
    "set the histogram bins using the specified values";

PyObject *
gsl::histogram::ranges(PyObject *, PyObject * args) {
    // the arguments
    PyObject * points;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_ranges",
                                  &PyCapsule_Type, &capsule,
                                  &PyTuple_Type, &points);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));
    // build the range array
    size_t size = PyTuple_Size(points);
    double * ranges = new double[size];
    // transfer the values
    for (size_t i = 0; i < size; i++) {
        ranges[i] = PyFloat_AsDouble(PyTuple_GET_ITEM(points, i));
    }
    // if anything went wrong
    if (PyErr_Occurred()) {
        // deallocate the range array
        delete [] ranges;
        // and raise an exception
        return 0;
    }
    // adjust the bins
    gsl_histogram_set_ranges(h, ranges, size);

    // deallocate the range array
    delete [] ranges;
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// reset
const char * const gsl::histogram::reset__name__ = "histogram_reset";
const char * const gsl::histogram::reset__doc__ = "reset a histogram";

PyObject *
gsl::histogram::reset(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_reset", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));
    // fill it out
    gsl_histogram_reset(h);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// increment
const char * const gsl::histogram::increment__name__ = "histogram_increment";
const char * const gsl::histogram::increment__doc__ =
    "increment by one the bin that contains the given value";

PyObject *
gsl::histogram::increment(PyObject *, PyObject * args) {
    // the arguments
    double x;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!d:histogram_increment",
                                  &PyCapsule_Type, &capsule,
                                  &x);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));
    // increment it
    gsl_histogram_increment(h, x);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// accumulate
const char * const gsl::histogram::accumulate__name__ = "histogram_accumulate";
const char * const gsl::histogram::accumulate__doc__ =
    "add the given weight to the bin that contains the given value";

PyObject *
gsl::histogram::accumulate(PyObject *, PyObject * args) {
    // the arguments
    double x, weight;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!dd:histogram_accumulate",
                                  &PyCapsule_Type, &capsule,
                                  &x, &weight);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));
    // accumulate the weight
    gsl_histogram_accumulate(h, x, weight);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// fill
const char * const gsl::histogram::fill__name__ = "histogram_fill";
const char * const gsl::histogram::fill__doc__ =
    "increment my frequency counts using values for the given vector";

PyObject *
gsl::histogram::fill(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    PyObject * vCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_fill",
                                  &PyCapsule_Type, &capsule,
                                  &PyCapsule_Type, &vCapsule
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the histogram capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }
    // bail out if the value vector capsule is not valid
    if (!PyCapsule_IsValid(vCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h =
        static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));
    // get the values
    gsl_vector * v =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(vCapsule, gsl::vector::capsule_t));

    // fill it out
    for (size_t i=0; i < v->size; i++) {
        gsl_histogram_increment(h, gsl_vector_get(v, i));
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// clone
const char * const gsl::histogram::clone__name__ = "histogram_clone";
const char * const gsl::histogram::clone__doc__ = "build a clone of a histogram";

PyObject *
gsl::histogram::clone(PyObject *, PyObject * args) {
    // the arguments
    PyObject * sourceCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_clone",
                                  &PyCapsule_Type, &sourceCapsule
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(sourceCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule for source");
        return 0;
    }

    // get the histograms
    gsl_histogram * source =
        static_cast<gsl_histogram *>(PyCapsule_GetPointer(sourceCapsule, capsule_t));
    gsl_histogram * clone;
    // clone the histogram
    clone = gsl_histogram_clone(source);

    // wrap it in a capsule and return it
    return PyCapsule_New(clone, capsule_t, free);
}


// copy
const char * const gsl::histogram::copy__name__ = "histogram_copy";
const char * const gsl::histogram::copy__doc__ = "build a copy of a histogram";

PyObject *
gsl::histogram::copy(PyObject *, PyObject * args) {
    // the arguments
    PyObject * sourceCapsule;
    PyObject * destinationCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_copy",
                                  &PyCapsule_Type, &destinationCapsule,
                                  &PyCapsule_Type, &sourceCapsule
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(sourceCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule for source");
        return 0;
    }
    // bail out if the destination capsule is not valid
    if (!PyCapsule_IsValid(destinationCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule for destination");
        return 0;
    }

    // get the histograms
    gsl_histogram * source =
        static_cast<gsl_histogram *>(PyCapsule_GetPointer(sourceCapsule, capsule_t));
    gsl_histogram * destination =
        static_cast<gsl_histogram *>(PyCapsule_GetPointer(destinationCapsule, capsule_t));
    // copy the data
    gsl_histogram_memcpy(destination, source);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// vector
const char * const gsl::histogram::vector__name__ = "histogram_vector";
const char * const gsl::histogram::vector__doc__ =
    "increment my frequency counts using values for the given vector";

PyObject *
gsl::histogram::vector(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_vector", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the histogram capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h =
        static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    // build the vector
    gsl_vector * v;

    // make the vector
    v = gsl_vector_alloc(h->n);
    // copy the data
    for (size_t i=0; i < h->n; i++) {
        gsl_vector_set(v, i, gsl_histogram_get(h, i));
    }

    // return a capsule with the vector data
    return PyCapsule_New(v, gsl::vector::capsule_t, gsl::vector::free);
}


// find
const char * const gsl::histogram::find__name__ = "histogram_find";
const char * const gsl::histogram::find__doc__ =
    "return the index of the bin the contains the given value";

PyObject *
gsl::histogram::find(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!d:histogram_find",
                                  &PyCapsule_Type, &capsule,
                                  &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    // find the bin index and return it
    size_t index;
    // find the value
    gsl_histogram_find(h, value, &index);

    // and return the index
    return PyLong_FromSize_t(index);
}


// max
const char * const gsl::histogram::max__name__ = "histogram_max";
const char * const gsl::histogram::max__doc__ = "return the maximum upper range";

PyObject *
gsl::histogram::max(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_max", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    double max;
    // compute the maximum
    max = gsl_histogram_max(h);

    // find the maximum upper range and return it
    return PyFloat_FromDouble(max);
}


// min
const char * const gsl::histogram::min__name__ = "histogram_min";
const char * const gsl::histogram::min__doc__ = "return the minimum lower range";

PyObject *
gsl::histogram::min(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_min", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    double min;
    // compute the minimum
    min = gsl_histogram_min(h);

    // find the minimum upper range and return it
    return PyFloat_FromDouble(min);
}


// range
const char * const gsl::histogram::range__name__ = "histogram_range";
const char * const gsl::histogram::range__doc__ =
    "return the range that corresponds to the given bin";

PyObject *
gsl::histogram::range(PyObject *, PyObject * args) {
    // the arguments
    size_t index;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!k:histogram_range",
                                  &PyCapsule_Type, &capsule,
                                  &index);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    // the endpoints
    double lower, upper;
    // compute them
    gsl_histogram_get_range(h, index, &lower, &upper);

    // place the result in a doublet
    PyObject * range = PyTuple_New(2);
    PyTuple_SET_ITEM(range, 0, PyFloat_FromDouble(lower));
    PyTuple_SET_ITEM(range, 1, PyFloat_FromDouble(upper));

    // and return it
    return range;
}


// max_bin
const char * const gsl::histogram::max_bin__name__ = "histogram_max_bin";
const char * const gsl::histogram::max_bin__doc__ =
    "return the index of the bin where the maximum value is contained";

PyObject *
gsl::histogram::max_bin(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_max_bin", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    size_t bin;
    // compute the bin
    bin = gsl_histogram_max_bin(h);

    // return the index
    return PyLong_FromSize_t(bin);
}


// min_bin
const char * const gsl::histogram::min_bin__name__ = "histogram_min_bin";
const char * const gsl::histogram::min_bin__doc__ =
    "return the index of the bin where the minimum value is contained";

PyObject *
gsl::histogram::min_bin(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_min_bin", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    size_t bin;
    // compute the bin
    bin = gsl_histogram_min_bin(h);

    // return the index
    return PyLong_FromSize_t(bin);
}


// max_val
const char * const gsl::histogram::max_val__name__ = "histogram_max_val";
const char * const gsl::histogram::max_val__doc__ =
    "find the maximum value in the histogram";

PyObject *
gsl::histogram::max_val(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_max_val", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    double value;
    // find the maximum value
    value = gsl_histogram_max_val(h);

    // find the value and return it
    return PyFloat_FromDouble(value);
}


// min_val
const char * const gsl::histogram::min_val__name__ = "histogram_min_val";
const char * const gsl::histogram::min_val__doc__ =
    "find the minimum value in the histogram";

PyObject *
gsl::histogram::min_val(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_min_val", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    double value;
    // find the maximum value
    value = gsl_histogram_min_val(h);

    // return the value
    return PyFloat_FromDouble(value);
}


// mean
const char * const gsl::histogram::mean__name__ = "histogram_mean";
const char * const gsl::histogram::mean__doc__ =
    "compute the mean value of the contents of a histogram";

PyObject *
gsl::histogram::mean(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_mean", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    double value;
    // find the maximum value
    value = gsl_histogram_mean(h);

    // return the value
    return PyFloat_FromDouble(value);
}


// sdev
const char * const gsl::histogram::sdev__name__ = "histogram_sdev";
const char * const gsl::histogram::sdev__doc__ =
    "compute the standard deviation of the contents of a histogram";

PyObject *
gsl::histogram::sdev(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_sdev", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    double value;
    // compute
    value = gsl_histogram_sigma(h);

    // compute the standard deviation and return it
    return PyFloat_FromDouble(value);
}


// sum
const char * const gsl::histogram::sum__name__ = "histogram_sum";
const char * const gsl::histogram::sum__doc__ =
    "compute the sum of the contents of a histogram";

PyObject *
gsl::histogram::sum(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:histogram_sum", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));

    double value;
    // compute
    value = gsl_histogram_sum(h);

    // compute the standard deviation and return it
    return PyFloat_FromDouble(value);
}


// access
const char * const gsl::histogram::get__name__ = "histogram_get";
const char * const gsl::histogram::get__doc__ = "get the value of a histogram element";

PyObject *
gsl::histogram::get(PyObject *, PyObject * args) {
    // the arguments
    size_t index;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!k:histogram_get", &PyCapsule_Type, &capsule, &index);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the histogram
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, capsule_t));
    // get the value
    double value;
    // compute
    value = gsl_histogram_get(h, index);

    // return the value
    return PyFloat_FromDouble(value);
}


// in-place operations
const char * const gsl::histogram::add__name__ = "histogram_add";
const char * const gsl::histogram::add__doc__ = "in-place addition of two histograms";

PyObject *
gsl::histogram::add(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_add",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the two histograms
    gsl_histogram * h1 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_histogram * h2 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(other, capsule_t));

    // perform the addition
    gsl_histogram_add(h1, h2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::histogram::sub__name__ = "histogram_sub";
const char * const gsl::histogram::sub__doc__ = "in-place subtraction of two histograms";

PyObject *
gsl::histogram::sub(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_sub",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the two histograms
    gsl_histogram * h1 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_histogram * h2 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(other, capsule_t));

    // perform the subtraction
    gsl_histogram_sub(h1, h2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::histogram::mul__name__ = "histogram_mul";
const char * const gsl::histogram::mul__doc__ = "in-place multiplication of two histograms";

PyObject *
gsl::histogram::mul(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_mul",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the two histograms
    gsl_histogram * h1 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_histogram * h2 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(other, capsule_t));

    // perform the multiplication
    gsl_histogram_mul(h1, h2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::histogram::div__name__ = "histogram_div";
const char * const gsl::histogram::div__doc__ = "in-place division of two histograms";

PyObject *
gsl::histogram::div(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:histogram_div",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the two histograms
    gsl_histogram * h1 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_histogram * h2 = static_cast<gsl_histogram *>(PyCapsule_GetPointer(other, capsule_t));

    // perform the division
    gsl_histogram_div(h1, h2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::histogram::shift__name__ = "histogram_shift";
const char * const gsl::histogram::shift__doc__ = "in-place addition of a constant to a histogram";

PyObject *
gsl::histogram::shift(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * self;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:histogram_shift", &PyCapsule_Type, &self, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the two histograms
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(self, capsule_t));

    // perform the shift
    gsl_histogram_shift(h, value);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::histogram::scale__name__ = "histogram_scale";
const char * const gsl::histogram::scale__doc__ = "in-place scaling of a histogram by a constant";

PyObject *
gsl::histogram::scale(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * self;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:histogram_scale", &PyCapsule_Type, &self, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid histogram capsule");
        return 0;
    }

    // get the two histograms
    gsl_histogram * h = static_cast<gsl_histogram *>(PyCapsule_GetPointer(self, capsule_t));

    // perform the scale
    gsl_histogram_scale(h, value);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// destructor
void
gsl::histogram::free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::histogram::capsule_t)) return;
    // get the histogram
    gsl_histogram * v =
        static_cast<gsl_histogram *>(PyCapsule_GetPointer(capsule, gsl::histogram::capsule_t));
    // std::cout << " gsl.histogram_free: histogram@" << v << std::endl;
    // deallocate
    gsl_histogram_free(v);
    // and return
    return;
}


// end of file
