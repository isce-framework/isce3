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
    
    def __cinit__(self, pyIH5File h5file):
        """
        Constructor that creates a C++ isce::product::Product objects and binds it to
        python instance.
        """
        self.c_product = new Product(deref(h5file.c_ih5file))
        self.__owner = True 

    def __dealloc__(self):
        if self.__owner:
            del self.c_product

    @property
    def complexImagery(self):
        """
        Get copy of complex imagery object.
        """
        im = pyComplexImagery()
        im.c_compleximagery = self.c_product.complexImagery()
        return im

    @property
    def metadata(self):
        """
        Get copy of metadata object.
        """
        meta = pyMetadata()
        meta.c_metadata = self.c_product.metadata()
        return meta

    @property
    def filename(self):
        """
        Get the filename of the HDF5 product file.
        """
        return str(self.c_product.filename())

# end of file
