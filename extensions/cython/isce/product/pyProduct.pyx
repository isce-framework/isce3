#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Product cimport Product

cdef class pyProduct:
    """
    Cython wrapper for isce::product::Product.

    Args:
        h5file (pyIH5File):                 IH5File for product.

    Return:
        None
    """
    # C++ class pointers
    cdef Product * c_product
    cdef bool __owner

    # Cython class members
    cdef pySwath py_swathA
    cdef pyMetadata py_metadata
    
    def __cinit__(self, pyIH5File h5file):
        """
        Constructor that creates a C++ isce::product::Product objects and binds it to
        python instance.
        """
        # Create the C++ Product class
        self.c_product = new Product(deref(h5file.c_ih5file))
        self.__owner = True 

        # Bind the C++ Metadata class to the Cython pyMetadata instance
        self.py_metadata.c_metadata = &self.c_product.metadata()
        self.py_metadata.__owner = False

        # Bind the C++ Swath class to the Cython pySwath instance
        self.py_swathA.c_swath = &self.c_product.swath(pyStringToBytes('A'))
        self.py_swathA.__owner = False

    def __dealloc__(self):
        if self.__owner:
            del self.c_product

    @property
    def metadata(self):
        """
        Get pyMetadata reference.
        """
        meta = pyMetadata.bind(self.py_metadata)
        return meta

    @property
    def swathA(self):
        """
        Get pySwath reference.
        """
        swath = pySwath.bind(self.py_swathA)
        return swath

    @property
    def filename(self):
        """
        Get the filename of the HDF5 product file.
        """
        return str(self.c_product.filename())

# end of file
