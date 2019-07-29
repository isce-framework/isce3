#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

# Exception types
# https://support.hdfgroup.org/HDF5/doc/cpplus_RM/class_h5_1_1_exception.html
class H5Exception(RuntimeError):
    pass
class H5AttributeIException(H5Exception):
    pass
class H5DataSetIException(H5Exception):
    pass
class H5DataSpaceIException(H5Exception):
    pass
class H5DataTypeIException(H5Exception):
    pass
class H5FileIException(H5Exception):
    pass
class H5GroupIException(H5Exception):
    pass
class H5IdComponentException(H5Exception):
    pass
class H5LibraryIException(H5Exception):
    pass
class H5LocationException(H5Exception):
    pass
class H5ObjHeaderIException(H5Exception):
    pass
class H5PropListIException(H5Exception):
    pass
class H5ReferenceException(H5Exception):
    pass

from cpython.ref cimport PyObject

cdef public PyObject* h5exception            = <PyObject*> H5Exception
cdef public PyObject* h5attributeiexception  = <PyObject*> H5AttributeIException
cdef public PyObject* h5datasetiexception    = <PyObject*> H5DataSetIException
cdef public PyObject* h5dataspaceiexception  = <PyObject*> H5DataSpaceIException
cdef public PyObject* h5datatypeiexception   = <PyObject*> H5DataTypeIException
cdef public PyObject* h5fileiexception       = <PyObject*> H5FileIException
cdef public PyObject* h5groupiexception      = <PyObject*> H5GroupIException
cdef public PyObject* h5idcomponentexception = <PyObject*> H5IdComponentException
cdef public PyObject* h5libraryiexception    = <PyObject*> H5LibraryIException
cdef public PyObject* h5locationexception    = <PyObject*> H5LocationException
cdef public PyObject* h5objheaderiexception  = <PyObject*> H5ObjHeaderIException
cdef public PyObject* h5proplistiexception   = <PyObject*> H5PropListIException
cdef public PyObject* h5referenceexception   = <PyObject*> H5ReferenceException

from libcpp.string cimport string
from IH5 cimport IH5File, IDataSet, IGroup

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


cdef class pyIGroup:
    """
    Cython wrapper for isce::io::IGroup.

    Args:
        None
    """
    cdef IGroup c_igroup

    def __cinit__(self):
        """
        Pre-constructor that creates a C++ isce::io::IGroup object and binds it to
        python instance.
        """
        self.c_igroup = IGroup()


cdef class pyIH5File:
    """
    Cython wrapper for isce::io::IH5File.

    Args:
        filename (str): Name of H5 file.
        access (Optional[int]): Access mode.
    """
    cdef IH5File * c_ih5file
    cdef bool __owner

    def __cinit__(self, filename, int access=0):
        """
        Pre-constructor that creates a C++ isce::io::IH5File object and binds it to 
        python instance.
        """
        # Convert the filename to a C++ string representation
        cdef string str_filename = pyStringToBytes(filename)

        # Create pointer to file
        self.c_ih5file = new IH5File(str_filename)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_ih5file

    def find(self, name, start, dtype='BOTH'):
        """
        Get a list of datasets with a given name relative to a starting path in HDF5 file.

        Args:
            name (str):             Dataset name to search for.
            start (str):            Starting path.
            dtype (Optional[str]):  Specify to return 'DATASET', 'GROUP', or 'BOTH'.

        Returns:
            datasets (list):        List of datasets found.
        """
        # Get vector of datasets
        cdef vector[string] dsetvec = self.c_ih5file.find(
            pyStringToBytes(name),
            pyStringToBytes(start),
            pyStringToBytes(dtype)
        )
        # Reformat as lists
        datasets = []
        cdef int i
        for i in range(dsetvec.size()):
            datasets.append(dsetvec[i])
        return datasets

    def openGroup(self, name):
        """
        Open an HDF5 group.

        Args:
            name (str):             Group name to search for.
            
        Returns:
            group (pyIGroup):       pyIGroup.
        """
        # Convert the name to a C++ string representation
        cdef string path = pyStringToBytes(name)

        # Open group
        new_group = pyIGroup()
        new_group.c_igroup = self.c_ih5file.openGroup(path)

        return new_group

# end of file
