// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// for the build system
#include <portinfo>
// external dependencies
#include <Python.h>
#include <pyre/mpi.h>

// the module method declarations
#include "communicators.h"
#include "exceptions.h"
#include "groups.h"
#include "metadata.h"
#include "ports.h"
#include "startup.h"


// put everything in my private namespace
namespace mpi {

    // the module method table
    PyMethodDef module_methods[] = {
        // module metadata
        // the copyright method
        { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
        // the version
        { version__name__, version, METH_VARARGS, version__doc__ },

        // init-fini
        { initialize__name__, initialize, METH_VARARGS, initialize__doc__ },
        { finalize__name__, finalize, METH_VARARGS, finalize__doc__ },

        // communicators
        { communicator::create__name__,
          communicator::create, METH_VARARGS, communicator::create__doc__ },
        { communicator::size__name__,
          communicator::size, METH_VARARGS, communicator::size__doc__ },
        { communicator::rank__name__,
          communicator::rank, METH_VARARGS, communicator::rank__doc__ },
        { communicator::barrier__name__,
          communicator::barrier, METH_VARARGS, communicator::barrier__doc__ },
        { communicator::bcast__name__,
          communicator::bcast, METH_VARARGS, communicator::bcast__doc__ },
        { communicator::sum__name__,
          communicator::sum, METH_VARARGS, communicator::sum__doc__ },
        { communicator::product__name__,
          communicator::product, METH_VARARGS, communicator::product__doc__ },
        { communicator::max__name__,
          communicator::max, METH_VARARGS, communicator::max__doc__ },
        { communicator::min__name__,
          communicator::min, METH_VARARGS, communicator::min__doc__ },
        { communicator::sum_all__name__,
          communicator::sum_all, METH_VARARGS, communicator::sum_all__doc__ },
        { communicator::product_all__name__,
          communicator::product_all, METH_VARARGS, communicator::product_all__doc__ },
        { communicator::max_all__name__,
          communicator::max_all, METH_VARARGS, communicator::max_all__doc__ },
        { communicator::min_all__name__,
          communicator::min_all, METH_VARARGS, communicator::min_all__doc__ },

        { cartesian::create__name__, cartesian::create, METH_VARARGS, cartesian::create__doc__ },
        { cartesian::coordinates__name__, cartesian::coordinates, METH_VARARGS,
          cartesian::coordinates__doc__ },

        // groups
        { group::isEmpty__name__, group::isEmpty, METH_VARARGS, group::isEmpty__doc__ },
        { group::create__name__, group::create, METH_VARARGS, group::create__doc__ },
        { group::size__name__, group::size, METH_VARARGS, group::size__doc__ },
        { group::rank__name__, group::rank, METH_VARARGS, group::rank__doc__ },
        { group::include__name__, group::include, METH_VARARGS, group::include__doc__ },
        { group::exclude__name__, group::exclude, METH_VARARGS, group::exclude__doc__ },
        { group::add__name__, group::add, METH_VARARGS, group::add__doc__ },
        { group::subtract__name__, group::subtract, METH_VARARGS, group::subtract__doc__ },
        { group::intersect__name__, group::intersect, METH_VARARGS, group::intersect__doc__ },

        // ports
        { port::sendBytes__name__, port::sendBytes, METH_VARARGS, port::sendBytes__doc__ },
        { port::recvBytes__name__, port::recvBytes, METH_VARARGS, port::recvBytes__doc__ },
        { port::sendString__name__, port::sendString, METH_VARARGS, port::sendString__doc__ },
        { port::recvString__name__, port::recvString, METH_VARARGS, port::recvString__doc__ },

        // sentinel
        {0, 0, 0, 0}
    };

    // the module documentation string
    const char * const __doc__ = "access to the MPI interface";

    // the module definition structure
    PyModuleDef module_definition = {
        // header
        PyModuleDef_HEAD_INIT,
        // the name of the module
        "mpi",
        // the module documentation string
        __doc__,
        // size of the per-interpreter state of the module; -1 if this state is global
        -1,
        // the methods defined in this module
        module_methods
    };

} // of namespace mpi


// initialization function for the module
// *must* be called PyInit_mpi
PyMODINIT_FUNC
PyInit_mpi()
{
    // mga - 20180328
    // N.B.: we used to initialize mpi as part of the import of this module
    // this seems to be a problem starting with openmpi 3.0

    // the reason is that openmpi no longer supports calling fork/exec of {mpiru} after
    // {MPI_Init} has been called by a process, which implies that we must delay initialization
    // until after the {launcher} has built the parallel machine

    // create the module
    PyObject * module = PyModule_Create(&mpi::module_definition);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return 0;
    }
    // otherwise, we have an initialized module
    mpi::registerExceptionHierarchy(module);

    // add the world communicator
    PyModule_AddObject(module, "world", mpi::communicator::world);

    // add some constants
    PyModule_AddObject(module, "undefined", PyLong_FromLong(MPI_UNDEFINED));
    PyModule_AddObject(module, "any_tag", PyLong_FromLong(MPI_ANY_TAG));
    PyModule_AddObject(module, "any_source", PyLong_FromLong(MPI_ANY_SOURCE));

    // and return the newly created module
    return module;
}

// end of file
