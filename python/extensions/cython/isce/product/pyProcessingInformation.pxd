#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from ProcessingInformation cimport ProcessingInformation

cdef class pyProcessingInformation:
    """
    Cython wrapper for isce::product::ProcessingInformation.

    Args:
        None

    Return:
        None
    """
    # C++ class pointers
    cdef ProcessingInformation * c_procinfo
    cdef bool __owner

# end of file
