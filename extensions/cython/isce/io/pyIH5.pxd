#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from IH5 cimport IH5File, IDataSet, IGroup

cdef class pyIDataSet:
    cdef IDataSet * c_idataset
    cdef bool __owner

cdef class pyIGroup:
    cdef IGroup c_igroup


cdef class pyIH5File:
    cdef IH5File * c_ih5file
    cdef bool __owner

# end of file
