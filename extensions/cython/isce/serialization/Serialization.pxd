#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018

from libcpp.string cimport string

from DateTime cimport DateTime
from Orbit cimport Orbit
from Ellipsoid cimport Ellipsoid
from EulerAngles cimport EulerAngles
from Metadata cimport Metadata
from Poly2d cimport Poly2d
from LUT1d cimport LUT1d
from IH5 cimport IH5File, IGroup

from Metadata cimport Metadata

# Wrapper around isce::core serialization defined in <isce/core/Serialization.h
cdef extern from "isce/core/Serialization.h" namespace "isce::core":

    # XML loading
    void load_archive[T](string metadata, char * objTag, T * obj)
    void load_archive_reference[T](string metadata, char * objTag, T & obj)

    # Load ellipsoid data
    void loadFromH5(IGroup & group, Ellipsoid & ellps)

    # Load orbit data
    void loadFromH5(IGroup & group, Orbit & orbit)

    # Load attitude data
    void loadFromH5(IGroup & group, EulerAngles & euler)

    # Load Poly2d
    void loadFromH5(IGroup & group, Poly2d & poly, string dtype)

    # Load LUT1d
    void loadFromH5(IGroup & group, LUT1d[double] & lut, string name_coords, string name_values)

    # Load metadata
    void loadFromH5(IGroup & group, Metadata & meta, string mode)

# Wrapper around isce::geometry serialization defined in <isce/geometry/Serialization.h
cdef extern from "isce/geometry/Serialization.h" namespace "isce::geometry":

    # XML loading
    void load_archive[T](string metadata, char * objTag, T * obj)

# Wrapper around isce::product serialization defined in <isce/product/Serialization.h
cdef extern from "isce/product/Serialization.h" namespace "isce::product":

    # Load metadata
    void loadFromH5(IGroup & group, Metadata & meta)

# end of file
