#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from EulerAngles cimport EulerAngles
from Orbit cimport Orbit
from ProcessingInformation cimport ProcessingInformation

cdef extern from "isce/product/Metadata.h" namespace "isce::product":

    # Metadata class
    cdef cppclass Metadata:

        # Constructors
        Metadata() except +

        # Attitude
        EulerAngles & attitude()
        void attitude(const EulerAngles &)

        # Orbit
        Orbit & orbit()
        void orbit(const Orbit &)

        # ProcessingInformation
        ProcessingInformation & procInfo()
        void procInfo(const ProcessingInformation &)

# end of file
