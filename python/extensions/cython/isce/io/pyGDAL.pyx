#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from GDAL cimport *

cdef class pyGDALDataset:
    """
    Basic Python wrapper for GDALDataset.

    Args:
        filename (str):         Filename on disk to read or create.
        access (Optional[int]): gdal.GA_ReadOnly or gdal.GA_Update
    """
    cdef GDALDataset * c_dataset
    cdef bool __owner

    def __cinit__(self, py_filename, GDALAccess access=GDALAccess.GA_ReadOnly):
        """
        C constructor.
        """
        cdef string filename = pyStringToBytes(py_filename)
        self.c_dataset = <GDALDataset *> GDALOpen(filename.c_str(), access)

    def __dealloc__(self):
        if self.__owner:
            del self.c_dataset


# end of file
