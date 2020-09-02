from libcpp.vector cimport vector
from libcpp.string cimport string
from Cartesian cimport cartesian_t, cartmat_t
from IH5 cimport IGroup

cdef extern from "isce3/core/Attitude.h" namespace "isce3::core":
    cdef cppclass Attitude:
        pass

# Wrapper around isce3::core serialization defined in <isce/core/Serialization.h
cdef extern from "isce3/core/Serialization.h" namespace "isce3::core":
    # Load attitude data
    void loadAttitude "loadFromH5" (IGroup & group, Attitude & attitude)

    # save attitude data
    void saveAttitude "saveToH5" (IGroup & group, Attitude & attitude)
