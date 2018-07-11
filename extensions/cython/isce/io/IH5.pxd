#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "isce/io/IH5.h" namespace "isce::io":

    # IDataSet class
    cdef cppclass IDataSet:

        # Constructors
        IDataSet() except +

        # Get the number of dimension of dataset or given attribute
        int getRank(const string &)

        # Get the total number of elements contained in dataset or given attribute
        int getNumElements(const string &)

        # Get the size of each dimension of the dataset or given attribute
        vector[int] getDimensions(const string &)

        # Get the H5 data type of the dataset or given attribute
        string getTypeClassStr(const string &)

        # Get the storage chunk size of the dataset
        vector[int] getChunkSize()

        # Get the number of bit used to store each dataset element
        int getNumBits(const string &)

    # IH5File class
    cdef cppclass IH5File:

        # Constructors
        IH5File(const string &) except +

        # Open a given dataset
        IDataSet openDataSet(const string &)

# end of file
