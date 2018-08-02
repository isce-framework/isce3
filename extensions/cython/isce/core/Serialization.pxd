#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018

from libcpp.string cimport string

from DateTime cimport DateTime
from Orbit cimport Orbit
from Ellipsoid cimport Ellipsoid
from Metadata cimport Metadata
from Poly2d cimport Poly2d
from IH5 cimport IH5File

# Wrapper around isce::core::load_archive in <isce/core/Serialization.h
cdef extern from "isce/core/Serialization.h" namespace "isce::core":
    void load_archive[T](string metadata, char * objTag, T * obj)
    void load_archive_reference[T](string metadata, char * objTag, T & obj)

    # Load ellipsoid data
    void load(IH5File & h5file, Ellipsoid & ellps)

    # Load orbit data
    void load(IH5File & h5file, Orbit & orbit, string orbit_type, DateTime & refEpoch)

    # Load Poly2d
    void load(IH5File & h5file, Poly2d & poly, string dtype)

    # Load metadata
    void load(IH5File & h5file, Metadata & meta, string mode)

# end of file
