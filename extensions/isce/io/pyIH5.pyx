#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from IH5 cimport IH5File, IDataSet

cdef class pyIDataSet:
    cdef IDataSet * c_idataset
    cdef bool __owner

    def __cinit__(self):
        self.c_idataset = new IDataSet()
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_idataset


cdef class pyIH5File:
    cdef IH5File * c_ih5file
    cdef bool __owner

    def __cinit__(self, py_filename, int access=0):

        # Convert the filename to a C++ string representation
        cdef string filename = pyStringToBytes(py_filename)

        # Create pointer to file
        self.c_ih5file = new IH5File(filename)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_ih5file

# end of file
