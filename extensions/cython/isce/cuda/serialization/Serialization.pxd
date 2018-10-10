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
from IH5 cimport IH5File, IGroup

from Radar cimport Radar

from ImageMode cimport ImageMode
from Metadata cimport Metadata
from Identification cimport Identification
from ComplexImagery cimport ComplexImagery

# Wrapper around isce::core serialization defined in <isce/core/Serialization.h
cdef extern from "isce/core/Serialization.h" namespace "isce::core":

    # XML loading
    void load_archive[T](string metadata, char * objTag, T * obj)
    void load_archive_reference[T](string metadata, char * objTag, T & obj)

    # Load ellipsoid data
    void loadFromH5(IGroup & group, Ellipsoid & ellps)

    # Load orbit data
    void loadFromH5(IGroup & group, Orbit & orbit, string orbit_type, DateTime & refEpoch)

    # Load Poly2d
    void loadFromH5(IGroup & group, Poly2d & poly, string dtype)

    # Load metadata
    void loadFromH5(IGroup & group, Metadata & meta, string mode)

# Wrapper around isce::geometry serialization defined in <isce/geometry/Serialization.h
cdef extern from "isce/geometry/Serialization.h" namespace "isce::geometry":

    # XML loading
    void load_archive[T](string metadata, char * objTag, T * obj)

# Wrapper around isce::radar serialization defined in <isce/radar/Serialization.h
cdef extern from "isce/radar/Serialization.h" namespace "isce::radar":

    # Load radar data
    void loadFromH5(IGroup & group, Radar & radar)

# Wrapper around isce::product serialization defined in <isce/product/Serialization.h
cdef extern from "isce/product/Serialization.h" namespace "isce::product":

    # Load image mode data
    void loadFromH5(IGroup & group, ImageMode & mode, const string &)

    # Load metadata
    void loadFromH5(IGroup & group, Metadata & meta)

    # Load identification
    void loadFromH5(IGroup & group, Identification & ident)

    # Load complex imagery
    void loadFromH5(IGroup & group, ComplexImagery & cpxImg)

# end of file
