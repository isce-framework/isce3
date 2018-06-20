#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from Ellipsoid cimport Ellipsoid
from Serialization cimport load_archive

cdef class pyEllipsoid:
    '''
    Python wrapper for isce::core::Ellipsoid

    Args:
        a (Optional[float]): Semi-major axis of Ellipsoid in m.
        e2 (Optional[float]): Eccentricity-squared.
    '''

    cdef Ellipsoid *c_ellipsoid
    cdef bool __owner

    def __cinit__(self):
        '''
        Pre-constructor that creates a C++ isce::core::Ellipsoid objects and binds it to python instance.
        '''

        self.c_ellipsoid = new Ellipsoid()
        self.__owner = True

    def __init__(self, a=0., e2=0.):
        self.a = a
        self.e2 = e2

    def __dealloc__(self):
        if self.__owner:
            del self.c_ellipsoid


    @staticmethod
    def bind(pyEllipsoid elp):
        '''
        Binds the current pyEllipsoid instance to another C++ Ellipsoid pointer.
        
        Args:
            elp (pyEllipsoid): Source of C++ Ellipsoid pointer. 
        '''
        new_elp = pyEllipsoid()
        del new_elp.c_ellipsoid
        new_elp.c_ellipsoid = elp.c_ellipsoid
        new_elp.__owner = False
        return new_elp


    @property
    def a(self):
        '''
        float: Semi-major axis of ellipsoid in meters.
        '''
        return self.c_ellipsoid.a()

    @property
    def b(self):
        '''
        float: Semi-minor axis of ellipsoid in meters.
        '''
        return self.c_ellipsoid.b()

    @a.setter
    def a(self, double a):
        '''
        Set the semi-major axis of ellipsoid in meters.

        Args:
            a (double) : Value of semi-major axis
        '''
        self.c_ellipsoid.a(a)


    @property
    def e2(self):
        '''
        float: Eccentricity-squared of ellipsoid.
        '''
        return self.c_ellipsoid.e2()


    @e2.setter
    def e2(self, double a):
        '''
        Set the eccentricity-squared of ellipsoid.
        '''
        self.c_ellipsoid.e2(a)

    def copyFrom(self, elp):
        '''
        Copy ellipsoid parameters with any class that has semi-major axis and eccentricity parameters.
        
        Args:
            elp (obj): Any object that has attributes a and e2.

        Returns:
            None
        '''
        # Replaces copy-constructor functionality
        try:
            self.a = elp.a
            self.e2 = elp.e2
        # Note: this allows for a dummy class object to be passed in that just has a and e2 as 
        # parameters!
        except: 
            print("Error: Object passed in to copy is incompatible with object of type " +
                  "pyEllipsoid.")

    def rEast(self, double lat):
        '''
        Prime Vertical Radius as a function of latitude. 

        Args:
            lat (float): Latitude in radians

        Returns:
            float: Prime Vertical radius in meters
            
        '''
        return self.c_ellipsoid.rEast(lat)

    def rNorth(self, double lat):
        '''
        Meridional radius as a function of latitude.

        Args:
            lat (float): Latitude in radians

        Returns:
            float: Meridional radius in meters
        '''
        return self.c_ellipsoid.rNorth(lat)

    def rDir(self, double lat, double hdg):
        '''
        Directional radius as a function of heading and latitude.

        Args:
            lat (float): Latitude in radians
            hdg (float): Heading in radians. Measured clockwise from North.

        Returns:
            float: Directional radius in meters.
        '''
        return self.c_ellipsoid.rDir(lat, hdg)


    def lonLatToXyz(self, list llh):
        '''
        Transform Lon/Lat/Hgt position to ECEF xyz coordinates.

        Args:
            llh (list): triplet of floats representing Lon (rad), Lat (rad) and hgt (m)

        Returns:
            list: triplet of floats representing ECEF coordinates in meters

        '''
        cdef cartesian_t inp
        cdef cartesian_t xyz

        for ii in range(3):
            inp[ii] = llh[ii]
        
        self.c_ellipsoid.lonLatToXyz(inp, xyz)
        out = [xyz[ii] for ii in range(3)]

        return out

    def xyzToLonLat(self, list xyz):
        '''
        Transform Lon/Lat/Hgt position to ECEF xyz coordinates.

        Args:
            xyz (list): triplet of floats representing ECEF coordinates in meters

        Returns:
            list : triplet of floats representing Lon (rad), Lat (rad) and hgt (m)

        '''
        cdef cartesian_t llh
        cdef cartesian_t inp

        for ii in range(3):
            inp[ii] = xyz[ii]
        
        self.c_ellipsoid.xyzToLonLat(inp,llh)
        out = [llh[ii] for ii in range(3)]
        
        return out

    def getImagingAnglesAtPlatform(self, list pos, list vel, list los):
        '''
        Compute azimuth angle and look angle at imaging platform.

        Args:
            pos (list): triplet of floats representing platform position in ECEF coordinates (m)
            vel (list): triplet of floats representing platform veloctity in ECEF coodinates (m/s)
            los (list): triplet of floats representing line-of-sight vector in ECEF coordiantes (m)

        Returns:
            (tuple): tuple containing: 
                * azi (float): Azimuth angle in radians. Measured anti-clockwise from North.
                * look (float): Look angle in radians. Measured w.r.t ellipsoid normal at platform.
        '''
        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef cartesian_t _los

        for ii in range(3):
            _pos[ii] = pos[ii]
            _vel[ii] = vel[ii]
            _los[ii] = los[ii]

        cdef double _azi = 0.
        cdef double _look = 0.

        self.c_ellipsoid.getImagingAnglesAtPlatform(_pos,_vel,_los,_azi,_look)
        return (_azi, _look)

    def TCNbasis(self, list pos, list vel):
        '''
        Compute TCN basis from platform position and velocity.

        Args:
            pos (list): triplet of floats representing ECEF position in meters
            vel (list): triplet of floats representing ECEF velocity in meters / sec

        Returns:
            (tuple): tuple containing:
                * that (list) - Tangential unit vector
                * chat (list) - Cross track unit vector
                * nhat (list) - Normal unit vector pointing downwards
        '''
        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef cartesian_t _t
        cdef cartesian_t _c
        cdef cartesian_t _n

        for i in range(3):
            _pos[i] = pos[i]
            _vel[i] = vel[i]
        
        self.c_ellipsoid.TCNbasis(_pos,_vel,_t,_c,_n)
        
        that = [_t[ii] for ii in range(3)]
        chat = [_c[ii] for ii in range(3)]
        nhat = [_n[ii] for ii in range(3)]

        return (that, chat, nhat)

    def archive(self, metadata):
        '''
        Load a string into ellipsoid object from a cereal archive.

        Args:
            metadata (str): Serialized XML corresponding to Ellipsoid.

        Returns:
            None
        '''
        load_archive[Ellipsoid](pyStringToBytes(metadata),
                                'Ellipsoid',
                                self.c_ellipsoid)

