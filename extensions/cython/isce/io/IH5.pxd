#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "H5Error.h":
    cdef void translateH5Exception();

cdef extern from "hdf5.h":
    # Basic types
    ctypedef long int hid_t


cdef extern from "isce/io/IH5.h" namespace "isce::io":

    # IDataSet class
    cdef cppclass IDataSet:

        # Constructors
        IDataSet() except +translateH5Exception

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

    # IGroup class
    cdef cppclass IGroup:

        # Constructors
        IGroup() except +translateH5Exception

        IGroup(hid_t & group) except +translateH5Exception

        # Open a given dataset
        IDataSet openDataSet(const string & name)

        # Open a given group
        IGroup openGroup(const string & name)

        # Find datasets with a given name
        vector[string] find(const string name, const string start, const string dtype,
                            const string returnedPath)



    # IH5File class
    cdef cppclass IH5File:

        # Constructors
        IH5File(const string & filename) except +translateH5Exception

        # Open a given dataset
        IDataSet openDataSet(const string & name)

        # Open a given group
        IGroup openGroup(const string & name)

        # Find datasets with a given name
        vector[string] find(const string name, const string start, const string dtype)

# end of file
