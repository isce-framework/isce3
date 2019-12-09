// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <sstream>
#include <cstdio>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics_double.h>

#include "vector.h"
#include "capsules.h"


// construction
const char * const gsl::vector::alloc__name__ = "vector_alloc";
const char * const gsl::vector::alloc__doc__ = "allocate a vector";

PyObject *
gsl::vector::alloc(PyObject *, PyObject * args) {
    // place holders for the python arguments
    size_t shape;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "k:vector_alloc", &shape);
    // if something went wrong
    if (!status) return 0;

    // allocate a vector
    gsl_vector * v = gsl_vector_alloc(shape);
    // std::cout << " gsl.vector_allocate: vector@" << v << ", size=" << shape << std::endl;

    // wrap it in a capsule and return it
    return PyCapsule_New(v, capsule_t, free);
}


// view construction
const char * const gsl::vector::view_alloc__name__ = "vector_view_alloc";
const char * const gsl::vector::view_alloc__doc__ = "allocate a vector view";

PyObject *
gsl::vector::view_alloc(PyObject *, PyObject * args) {
    // place holders for the python arguments
    size_t origin;
    size_t shape;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args,
                                  "O!kk:vector_view_alloc",
                                  &PyCapsule_Type, &capsule,
                                  &origin, &shape);
    // if something went wrong
    if (!status) return 0;
    // bail out if the vector capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // build the vector view
    gsl_vector_view * view = new gsl_vector_view(gsl_vector_subvector(v, origin, shape));

    // the caller expects a tuple
    PyObject * result = PyTuple_New(2);
    // the zeroth entry is the capsule
    PyTuple_SET_ITEM(result, 0, PyCapsule_New(view, view_t, freeview));
    // followed by a pointer to the view data
    // N.B.: don't attempt to deallocate this one...
    PyTuple_SET_ITEM(result, 1, PyCapsule_New(&(view->vector), capsule_t, 0));

    // all done
    return result;
}


// initialization
const char * const gsl::vector::zero__name__ = "vector_zero";
const char * const gsl::vector::zero__doc__ = "zero out the elements of a vector";

PyObject *
gsl::vector::zero(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_zero", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.vector_zero: vector@" << v << std::endl;
    // zero it out
    gsl_vector_set_zero(v);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::vector::fill__name__ = "vector_fill";
const char * const gsl::vector::fill__doc__ = "set all elements of a vector to a value";

PyObject *
gsl::vector::fill(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:vector_fill", &PyCapsule_Type, &capsule, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.vector_fill: vector@" << v << ", value=" << value << std::endl;
    // fill it out
    gsl_vector_set_all(v, value);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// basis
const char * const gsl::vector::basis__name__ = "vector_basis";
const char * const gsl::vector::basis__doc__ = "build a basis vector";

PyObject *
gsl::vector::basis(PyObject *, PyObject * args) {
    // the arguments
    size_t index;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!k:vector_basis", &PyCapsule_Type, &capsule, &index);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout << " gsl.vector_basis: vector@" << v << ", index=" << index << std::endl;
    // fill it out
    gsl_vector_set_basis(v, index);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// copy
const char * const gsl::vector::copy__name__ = "vector_copy";
const char * const gsl::vector::copy__doc__ = "build a copy of a vector";

PyObject *
gsl::vector::copy(PyObject *, PyObject * args) {
    // the arguments
    PyObject * sourceCapsule;
    PyObject * destinationCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_copy",
                                  &PyCapsule_Type, &destinationCapsule,
                                  &PyCapsule_Type, &sourceCapsule
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(sourceCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
        return 0;
    }
    // bail out if the destination capsule is not valid
    if (!PyCapsule_IsValid(destinationCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for destination");
        return 0;
    }

    // get the vectors
    gsl_vector * source =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(sourceCapsule, capsule_t));
    gsl_vector * destination =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(destinationCapsule, capsule_t));
    // copy the data
    gsl_vector_memcpy(destination, source);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// tuple
const char * const gsl::vector::tuple__name__ = "vector_tuple";
const char * const gsl::vector::tuple__doc__ = "build a tuple out of a vector";

PyObject *
gsl::vector::tuple(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!:vector_tuple",
                                  &PyCapsule_Type, &capsule
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // get the shape
    size_t s = v->size;

    // we return a tuple
    PyObject * result = PyTuple_New(s);
    // go through the elements
    for (size_t slot=0; slot<s; ++slot) {
        // grab the value, turn it into a float and attach it
        PyTuple_SET_ITEM(result, slot, PyFloat_FromDouble(gsl_vector_get(v, slot)));
    }

    // return the result
    return result;
}


// read
const char * const gsl::vector::read__name__ = "vector_read";
const char * const gsl::vector::read__doc__ = "read the values of a vector from a binary file";

PyObject *
gsl::vector::read(PyObject *, PyObject * args) {
    // the arguments
    char * filename;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!s:vector_read", &PyCapsule_Type, &capsule, &filename);

    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
        return 0;
    }

    // attempt to open the stream
    std::FILE * stream = std::fopen(filename, "rb");
    // bail out if something went wrong
    if (!stream) {
        PyErr_SetString(PyExc_IOError, "could not open file for reading");
        return 0;
    }

    // get the vector
    gsl_vector * v =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // read the data
    gsl_vector_fread(stream, v);
    // close the file
    std::fclose(stream);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// write
const char * const gsl::vector::write__name__ = "vector_write";
const char * const gsl::vector::write__doc__ = "write the values of a vector to a binary file";

PyObject *
gsl::vector::write(PyObject *, PyObject * args) {
    // the arguments
    char * filename;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!s:vector_write", &PyCapsule_Type, &capsule, &filename);

    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
        return 0;
    }

    // attempt to open the stream
    FILE * stream = std::fopen(filename, "wb");
    // bail out if something went wrong
    if (!stream) {
        PyErr_SetString(PyExc_IOError, "could not open file for writing");
        return 0;
    }

    // get the vector
    gsl_vector * v =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // write the data
    gsl_vector_fwrite(stream, v);

    // close the file
    std::fclose(stream);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// scanf
const char * const gsl::vector::scanf__name__ = "vector_scanf";
const char * const gsl::vector::scanf__doc__ = "read the values of a vector from a text file";

PyObject *
gsl::vector::scanf(PyObject *, PyObject * args) {
    // the arguments
    char * filename;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!s:vector_scanf", &PyCapsule_Type, &capsule, &filename);

    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
        return 0;
    }

    // attempt to open the stream
    std::FILE * stream = std::fopen(filename, "r");
    // bail out if something went wrong
    if (!stream) {
        PyErr_SetString(PyExc_IOError, "could not open file for reading");
        return 0;
    }

    // get the vector
    gsl_vector * v =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // read the data
    gsl_vector_fscanf(stream, v);
    // close the file
    std::fclose(stream);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// printf
const char * const gsl::vector::printf__name__ = "vector_printf";
const char * const gsl::vector::printf__doc__ = "write the values of a vector to a file";

PyObject *
gsl::vector::printf(PyObject *, PyObject * args) {
    // the arguments
    char * filename;
    char * format;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args,
                                  "O!ss:vector_printf",
                                  &PyCapsule_Type,
                                  &capsule,
                                  &filename,
                                  &format);

    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
        return 0;
    }

    // attempt to open the stream
    FILE * stream = std::fopen(filename, "w");
    // bail out if something went wrong
    if (!stream) {
        PyErr_SetString(PyExc_IOError, "could not open file for writing");
        return 0;
    }

    // get the vector
    gsl_vector * v =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // write the data
    gsl_vector_fprintf(stream, v, format);

    // close the file
    std::fclose(stream);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// access
const char * const gsl::vector::get__name__ = "vector_get";
const char * const gsl::vector::get__doc__ = "get the value of a vector element";

PyObject *
gsl::vector::get(PyObject *, PyObject * args) {
    // the arguments
    long index;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!l:vector_get", &PyCapsule_Type, &capsule, &index);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // reflect negative indices about the end of the vector
    if (index < 0) index += v->size;
    // convert to an unsigned value
    size_t i = index;
    // bounds check index 1
    if (i >= v->size) {
        // build an error message
        std::stringstream msg;
        msg << "vector index " << index << " out of range";
        // register the error
        PyErr_SetString(PyExc_IndexError, msg.str().c_str());
        // and raise the exception
        return 0;
    }

    // get the value
    double value = gsl_vector_get(v, i);
    // std::cout
        // << " gsl.vector_get: vector@" << v << ", index=" << index << ", value=" << value
        // << std::endl;

    // return the value
    return PyFloat_FromDouble(value);
}


const char * const gsl::vector::set__name__ = "vector_set";
const char * const gsl::vector::set__doc__ = "set the value of a vector element";

PyObject *
gsl::vector::set(PyObject *, PyObject * args) {
    // the arguments
    long index;
    double value;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!ld:vector_set",
                                  &PyCapsule_Type, &capsule, &index, &value);
    // bail out if something went wrong with the argument unpacking
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout
        // << " gsl.vector_set: vector@" << v << ", index=" << index << ", value=" << value
        // << std::endl;

    // reflect negative indices about the end of the vector
    if (index < 0) index += v->size;
    // convert to an unsigned value
    size_t i = index;
    // bounds check index 1
    if (i >= v->size) {
        // build an error message
        std::stringstream msg;
        msg << "vector index " << index << " out of range";
        // register the error
        PyErr_SetString(PyExc_IndexError, msg.str().c_str());
        // and raise the exception
        return 0;
    }
    // set the value
    gsl_vector_set(v, i, value);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::vector::contains__name__ = "vector_contains";
const char * const gsl::vector::contains__doc__ = "check whether a given value appears in vector";

PyObject *
gsl::vector::contains(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:vector_contains", &PyCapsule_Type, &capsule, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // std::cout
        // << " gsl.vector_contains: vector@" << v << ", index=" << index << ", value=" << value
        // << std::endl;

    // the answer
    PyObject * result = Py_False;

    // loop over the elements
    for (size_t index=0; index < v->size; index++) {
        // if i have a match
        if (value == gsl_vector_get(v, index)) {
            // update the answer
            result = Py_True;
            // and bail
            break;
        }
    }

    // return the answer
    Py_INCREF(result);
    return result;
}


// minima and maxima
const char * const gsl::vector::max__name__ = "vector_max";
const char * const gsl::vector::max__doc__ = "find the largest value contained";

PyObject *
gsl::vector::max(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_max", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // compute the max
    double value = gsl_vector_max(v);
    // std::cout << " gsl.vector_max: vector@" << v << ", value=" << value << std::endl;

    // return the value
    return PyFloat_FromDouble(value);
}


const char * const gsl::vector::min__name__ = "vector_min";
const char * const gsl::vector::min__doc__ = "find the smallest value contained";

PyObject *
gsl::vector::min(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_min", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // compute
    double value = gsl_vector_min(v);
    // std::cout << " gsl.vector_max: vector@" << v << ", value=" << value << std::endl;

    // return the value
    return PyFloat_FromDouble(value);
}


const char * const gsl::vector::minmax__name__ = "vector_minmax";
const char * const gsl::vector::minmax__doc__ =
    "find both the smallest and the largest value contained";

PyObject *
gsl::vector::minmax(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_minmax", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    double small, large;
    gsl_vector_minmax(v, &small, &large);
    // std::cout
        // << " gsl.vector_max: vector@" << v << ", min=" << small << ", max=" << large
        // << std::endl;

    // build the answer
    PyObject * answer = PyTuple_New(2);
    PyTuple_SET_ITEM(answer, 0, PyFloat_FromDouble(small));
    PyTuple_SET_ITEM(answer, 1, PyFloat_FromDouble(large));
    // and return
    return answer;
}


// equality
const char * const gsl::vector::equal__name__ = "vector_equal";
const char * const gsl::vector::equal__doc__ = "check two vectors for equality";

PyObject *
gsl::vector::equal(PyObject *, PyObject * args) {
    // the arguments
    PyObject * leftCapsule;
    PyObject * rightCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_equal",
                                  &PyCapsule_Type, &rightCapsule,
                                  &PyCapsule_Type, &leftCapsule
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the left capsule is not valid
    if (!PyCapsule_IsValid(leftCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for the left operand");
        return 0;
    }
    // bail out if the right capsule is not valid
    if (!PyCapsule_IsValid(rightCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for the right operand");
        return 0;
    }

    // get the vectors
    gsl_vector * left =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(leftCapsule, capsule_t));
    gsl_vector * right =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(rightCapsule, capsule_t));

    // the answer
    PyObject * answer = gsl_vector_equal(left, right) ? Py_True : Py_False;

    // return
    Py_INCREF(answer);
    return answer;
}


// in-place operations
const char * const gsl::vector::add__name__ = "vector_add";
const char * const gsl::vector::add__doc__ = "in-place addition of two vectors";

PyObject *
gsl::vector::add(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_add",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    gsl_vector * v1 = static_cast<gsl_vector *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_vector * v2 = static_cast<gsl_vector *>(PyCapsule_GetPointer(other, capsule_t));
    // std::cout << " gsl.vector_add: vector@" << v1 << ", vector@" << v2 << std::endl;

    // perform the addition
    gsl_vector_add(v1, v2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::vector::sub__name__ = "vector_sub";
const char * const gsl::vector::sub__doc__ = "in-place subtraction of two vectors";

PyObject *
gsl::vector::sub(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_sub",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    gsl_vector * v1 = static_cast<gsl_vector *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_vector * v2 = static_cast<gsl_vector *>(PyCapsule_GetPointer(other, capsule_t));
    // std::cout << " gsl.vector_sub: vector@" << v1 << ", vector@" << v2 << std::endl;

    // perform the subtraction
    gsl_vector_sub(v1, v2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::vector::mul__name__ = "vector_mul";
const char * const gsl::vector::mul__doc__ = "in-place multiplication of two vectors";

PyObject *
gsl::vector::mul(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_mul",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    gsl_vector * v1 = static_cast<gsl_vector *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_vector * v2 = static_cast<gsl_vector *>(PyCapsule_GetPointer(other, capsule_t));
    // std::cout << " gsl.vector_mul: vector@" << v1 << ", vector@" << v2 << std::endl;

    // perform the multiplication
    gsl_vector_mul(v1, v2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::vector::div__name__ = "vector_div";
const char * const gsl::vector::div__doc__ = "in-place division of two vectors";

PyObject *
gsl::vector::div(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_div",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    gsl_vector * v1 = static_cast<gsl_vector *>(PyCapsule_GetPointer(self, capsule_t));
    gsl_vector * v2 = static_cast<gsl_vector *>(PyCapsule_GetPointer(other, capsule_t));
    // std::cout << " gsl.vector_div: vector@" << v1 << ", vector@" << v2 << std::endl;

    // perform the division
    gsl_vector_div(v1, v2);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::vector::shift__name__ = "vector_shift";
const char * const gsl::vector::shift__doc__ = "in-place addition of a constant to a vector";

PyObject *
gsl::vector::shift(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * self;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:vector_shift", &PyCapsule_Type, &self, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(self, capsule_t));
    // std::cout << " gsl.vector_shift: vector@" << v << ", value=" << value << std::endl;

    // perform the shift
    gsl_vector_add_constant(v, value);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const gsl::vector::scale__name__ = "vector_scale";
const char * const gsl::vector::scale__doc__ = "in-place scaling of a vector by a constant";

PyObject *
gsl::vector::scale(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * self;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:vector_scale", &PyCapsule_Type, &self, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(self, capsule_t));
    // std::cout << " gsl.vector_scale: vector@" << v << ", value=" << value << std::endl;

    // perform the scale
    gsl_vector_scale(v, value);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// statistics
// sort
const char * const gsl::vector::sort__name__ = "vector_sort";
const char * const gsl::vector::sort__doc__ = "in-place sort the elements of a vector";

PyObject *
gsl::vector::sort(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_sort", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // sort it
    gsl_sort_vector(v);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// sortIndex
const char * const gsl::vector::sortIndex__name__ = "vector_sortIndex";
const char * const gsl::vector::sortIndex__doc__ =
    "construct the permutation that would sort the elements of a vector";

PyObject *
gsl::vector::sortIndex(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_sortIndex", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    // allocate the permutation
    gsl_permutation * p = gsl_permutation_alloc(v->size);

    // sort it
    gsl_sort_vector_index(p, v);

    // return a permutation capsule
    return PyCapsule_New(p, gsl::permutation::capsule_t, gsl::permutation::free);
}


// mean
const char * const gsl::vector::mean__name__ = "vector_mean";
const char * const gsl::vector::mean__doc__ = "compute the mean of the elements of a vector";

PyObject *
gsl::vector::mean(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    PyObject * weights;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O:vector_mean",
                                  &PyCapsule_Type, &capsule, &weights);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // the answer
    double mean;
    // if no weights were given
    if (weights == Py_None) {
        // compute the mean
        mean = gsl_stats_mean(v->data, v->stride, v->size);
    } else {
        // otherwise, check that {weights} is a vector capsule
        if (!PyCapsule_IsValid(weights, capsule_t)) {
            PyErr_SetString(PyExc_TypeError, "invalid vector capsule for the weights");
            return 0;
        }
        // extract the  vector of weights
        gsl_vector * w = static_cast<gsl_vector *>(PyCapsule_GetPointer(weights, capsule_t));
        // compute the weighted mean
        mean = gsl_stats_wmean(w->data, w->stride, v->data, v->stride, v->size);
    }
    // and return it
    return PyFloat_FromDouble(mean);
}


// median
const char * const gsl::vector::median__name__ = "vector_median";
const char * const gsl::vector::median__doc__ =
    "compute the median of the elements of a pre-sorted vector";

PyObject *
gsl::vector::median(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_median", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // compute the median
    double median = gsl_stats_median_from_sorted_data(v->data, v->stride, v->size);

    // and return it
    return PyFloat_FromDouble(median);
}


// variance
const char * const gsl::vector::variance__name__ = "vector_variance";
const char * const gsl::vector::variance__doc__ =
    "compute the variance of the elements of a vector";

PyObject *
gsl::vector::variance(PyObject *, PyObject * args) {
    // the arguments
    PyObject * mean;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!O:vector_variance", &PyCapsule_Type, &capsule, &mean);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // the answer
    double variance;
    // three cases
    if (mean == Py_None) {
        // {mean} is {None}: compute the mean on the fly
        variance = gsl_stats_variance(v->data, v->stride, v->size);
    } else if (PyFloat_Check(mean)) {
        // {mean} is a float: use it
        variance = gsl_stats_variance_m(v->data, v->stride, v->size, PyFloat_AsDouble(mean));
    } else {
        // {mean} is anything else: raise an exception
        PyErr_SetString(PyExc_TypeError, "{mean} must be a float");
        return 0;
    }
    // and return the variance
    return PyFloat_FromDouble(variance);
}


// sdev
const char * const gsl::vector::sdev__name__ = "vector_sdev";
const char * const gsl::vector::sdev__doc__ =
    "compute the standard deviation of the elements of a vector";

PyObject *
gsl::vector::sdev(PyObject *, PyObject * args) {
    // the arguments
    PyObject * mean;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!O:vector_sdev", &PyCapsule_Type, &capsule, &mean);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // the answer
    double sdev;
    // three cases
    if (mean == Py_None) {
        // {mean} is {None}: compute the mean on the fly
        sdev = gsl_stats_sd(v->data, v->stride, v->size);
    } else if (PyFloat_Check(mean)) {
        // {mean} is a float: use it
        sdev = gsl_stats_sd_m(v->data, v->stride, v->size, PyFloat_AsDouble(mean));
    } else {
        // {mean} is anything else: raise an exception
        PyErr_SetString(PyExc_TypeError, "{mean} must be a float");
        return 0;
    }
    // and return the sdev
    return PyFloat_FromDouble(sdev);
}


// destructors
void
gsl::vector::free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::vector::capsule_t)) return;
    // get the vector
    gsl_vector * v =
        static_cast<gsl_vector *>(PyCapsule_GetPointer(capsule, gsl::vector::capsule_t));
    // std::cout << " gsl.vector_free: vector@" << v << std::endl;
    // deallocate
    gsl_vector_free(v);
    // and return
    return;
}


void
gsl::vector::freeview(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::vector::view_t)) return;
    // get the vector view
    gsl_vector_view * v =
        static_cast<gsl_vector_view *>(PyCapsule_GetPointer(capsule, gsl::vector::view_t));
    // deallocate
    delete v;
    // and return
    return;
}


// end of file
