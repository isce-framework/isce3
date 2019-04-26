from cpython.ref cimport PyObject
from Error cimport *

class Error(RuntimeError):
    pass

cdef public PyObject* errobj = <PyObject*> Error
