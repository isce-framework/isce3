#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from IH5 cimport IH5File, IDataSet

cdef class pyIDataSet:
    """
    Cython wrapper for isce::io::IDataSet.

    Args:
        None
    """
    cdef IDataSet * c_idataset
    cdef bool __owner

    def __cinit__(self):
        """
        Pre-constructor that creates a C++ isce::io::IDataSet object and binds it to 
        python instance.
        """
        self.c_idataset = new IDataSet()
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_idataset


cdef class pyIH5File:
    """
    Cython wrapper for isce::io::IH5File.

    Args:
        py_filename (str): Name of H5 file.
        access (Optional[int]): Access mode.
    """
    cdef IH5File * c_ih5file
    cdef bool __owner

    def __cinit__(self, py_filename, int access=0):
        """
        Pre-constructor that creates a C++ isce::io::IH5File object and binds it to 
        python instance.
        """
        # Convert the filename to a C++ string representation
        cdef string filename = pyStringToBytes(py_filename)

        # Create pointer to file
        self.c_ih5file = new IH5File(filename)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_ih5file

# end of file
