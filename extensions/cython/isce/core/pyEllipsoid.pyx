#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from Ellipsoid cimport Ellipsoid
from numpy cimport ndarray
import numpy as np
cimport numpy as np

cdef class pyEllipsoid:
    '''
    Python wrapper for isce::core::Ellipsoid.
    By default returns a WGS84 Ellipsoid.

    Args:
        a (float, optional): Semi-major axis of Ellipsoid in m.
        e2 (float, optional): Eccentricity-squared.
    '''

    cdef Ellipsoid *c_ellipsoid
    cdef bool __owner

    def __cinit__(self):
        '''
        Pre-constructor that creates a C++ isce::core::Ellipsoid objects and binds it to python instance.
        '''

        self.c_ellipsoid = new Ellipsoid()
        self.__owner = True

    def __init__(self, a=6378137., e2=.0066943799901):
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
            elp (:obj:`pyEllipsoid`): Source of C++ Ellipsoid pointer. 
        '''
        new_elp = pyEllipsoid()
        del new_elp.c_ellipsoid
        new_elp.c_ellipsoid = elp.c_ellipsoid
        new_elp.__owner = False
        return new_elp

    @staticmethod
    cdef cbind(Ellipsoid elp):
        '''
        Creates a new pyEllipsoid instance from a C++ Ellipsoid instance.
        
        Args:
            elp (Ellipsoid): C++ Ellipsoid instance.
        '''
        new_elp = pyEllipsoid()
        del new_elp.c_ellipsoid
        new_elp.c_ellipsoid = new Ellipsoid(elp)
        new_elp.__owner = True
        return new_elp

    @property
    def a(self):
        '''
        Semi-major axis of ellipsoid.

        Returns:
            float: Semi-major axis of ellipsoid in meters.
        '''
        return self.c_ellipsoid.a()

    @property
    def b(self):
        '''
        Semi-minor axis of ellipsoid.

        Returns:
            float: Semi-minor axis of ellipsoid in meters.
        '''
        return self.c_ellipsoid.b()

    @a.setter
    def a(self, double a):
        '''
        Set the semi-major axis of ellipsoid in meters.

        Args:
            a (float) : Value of semi-major axis
        '''
        self.c_ellipsoid.a(a)


    @property
    def e2(self):
        '''
        Eccentricity-squared of ellipsoid.

        Returns:
            float: Eccentricity-squared of ellipsoid.
        '''
        return self.c_ellipsoid.e2()


    @e2.setter
    def e2(self, double a):
        '''
        Set the eccentricity-squared of ellipsoid.

        Args:
            a (float): Value to eccentricity-squared
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
 
    def rEast(self, lat):
        '''
        Prime Vertical Radius as a function of latitude. 

        Args:
            lat (float or :obj:`numpy.ndarray`): Latitude in radians. Can be constant or 1D array.

        Returns:
            float or :obj:`numpy.ndarray`: Prime Vertical radius in meters
            
        '''
        #Single value
        if np.isscalar(lat):
            return self.c_ellipsoid.rEast(lat)

        #For loop over array
        lat = np.atleast_1d(lat)
        cdef unsigned long nPts = lat.shape[0]
        cdef unsigned long ii
        res = np.empty(nPts, dtype=np.double)
        cdef double[:] resview = res

        for ii in range(nPts):
            resview[ii] =  self.c_ellipsoid.rEast(lat[ii])

        return res


    def rNorth(self, lat):
        '''
        Meridional radius as a function of latitude.

        Args:
            lat (float or :obj:`numpy.ndarray`): Latitude in radians. Can be constant or 1D array.

        Returns:
            float or :obj:`numpy.ndarray`: Meridional radius in meters
        '''
        #Single value
        if np.isscalar(lat):
            return self.c_ellipsoid.rNorth(lat)

        #For loop over array
        lat = np.atleast_1d(lat)
        cdef unsigned long nPts = lat.shape[0]
        cdef unsigned long ii
        res = np.empty(nPts, dtype=np.double)
        cdef double[:] resview = res
        
        for ii in range(nPts):
            resview[ii] = self.c_ellipsoid.rNorth(lat[ii])

        return res

    def rDir(self, lat, hdg):
        '''
        Directional radius as a function of heading and latitude.

        Note:
            lat and hdg should be of same size.

        Args:
            lat (float or :obj:`numpy.ndarray`): Latitude in radians. Can be constant or 1D array.
            hdg (float or :obj:`numpy.ndarray`): Heading in radians. Measured clockwise from North.

        Returns:
            float or :obj:`numpy.ndarray`: Directional radius in meters.
        '''

        if np.isscalar(lat) and np.isscalar(hdg):
            return self.c_ellipsoid.rDir(lat, hdg)

        lat = np.atleast_1d(lat)
        hdg = np.atleast_1d(hdg)
        assert(lat.shape[0] == hdg.shape[0])

        cdef unsigned long nPts = lat.shape[0]
        cdef unsigned long ii
        res = np.empty(nPts, dtype=np.double)
        cdef double[:] resview = res
        for ii in range(nPts):
            resview[ii] = self.c_ellipsoid.rDir(lat[ii], hdg[ii])

        return res

    
    def lonLatToXyz(self, llh):
        '''
        Transform Lon/Lat/Hgt position to ECEF xyz coordinates.

        Args:
            llh (:obj:`numpy.ndarray`): triplet of floats representing Lon (rad), Lat (rad) and hgt (m).
                Can be of shape (3,) or (n,3).

        Returns:
            :obj:`numpy.ndarray`: triplet of floats representing ECEF coordinates in meters

        '''
        llh = np.atleast_2d(llh)
        cdef unsigned long nPts = llh.shape[0]
        res = np.empty((nPts,3), dtype=np.double)
        cdef double[:,:] resview = res

        cdef unsigned long ii
        cdef int jj
        cdef cartesian_t inp
        cdef cartesian_t xyz
        for ii in range(nPts):
            for jj in range(3):
                inp[jj] = llh[ii,jj]
            self.c_ellipsoid.lonLatToXyz(inp,xyz)
            for jj in range(3):
                resview[ii,jj] = xyz[jj]
        
        return np.squeeze(res)

    def xyzToLonLat(self, xyz):
        '''
        Transform Lon/Lat/Hgt position to ECEF xyz coordinates.

        Args:
            xyz (:obj:`numpy.ndarray`): triplet of floats representing ECEF coordinates in meters.
                Can be of shape (3,) or (n,3).

        Returns:
            :obj:`numpy.ndarray`: triplet of floats representing Lon (rad), Lat (rad) and hgt (m)

        '''
        xyz = np.atleast_2d(xyz)
        cdef unsigned long nPts = xyz.shape[0]
        res = np.empty((nPts,3), dtype=np.double)
        cdef double[:,:] resview = res

        cdef unsigned long ii
        cdef int jj

        cdef cartesian_t llh
        cdef double[::1] llhview = <double[:3]>(&llh[0])

        cdef cartesian_t inp
        cdef double[::1] inpview = <double[:3]>(&inp[0])

        for ii in range(nPts):
            for jj in range(3):
                inpview[jj] = xyz[ii,jj]
            self.c_ellipsoid.xyzToLonLat(inp, llh)
            for jj in range(3):
                resview[ii,jj] = llhview[jj]
        
        return np.squeeze(res)

    def getImagingAnglesAtPlatform(self, pos, vel, los):
        '''
        Compute azimuth angle and look angle at imaging platform.

        Args:
            pos (:obj:`numpy.ndarray`): triplet of floats representing platform position in ECEF coordinates (m)
            vel (:obj:`numpy.ndarray`): triplet of floats representing platform veloctity in ECEF coodinates (m/s)
            los (:obj:`numpy.ndarray`): triplet of floats representing line-of-sight vector in ECEF coordiantes (m)

        Returns:
            (tuple): tuple containing:
                * azi (float): Azimuth angle in radians. Measured anti-clockwise from North.
                * look (float): Look angle in radians. Measured w.r.t ellipsoid normal at platform.
        '''
        cdef cartesian_t _pos
        cdef double[::1] _posview = <double[:3]>(&_pos[0])

        cdef cartesian_t _vel
        cdef double[::1] _velview = <double[:3]>(&_vel[0])

        cdef cartesian_t _los
        cdef double[::1] _losview = <double[:3]>(&_los[0])

        _posview[:] = pos[:]
        _velview[:] = vel[:]
        _losview[:] = los[:]

        cdef double _azi = 0.
        cdef double _look = 0.

        self.c_ellipsoid.getImagingAnglesAtPlatform(_pos,_vel,_los,_azi,_look)
        return (_azi, _look)

    def TCNbasis(self, pos, vel):
        '''
        Compute TCN basis from platform position and velocity.

        Args:
            pos (:obj:`numpy.ndarray`): triplet of floats representing ECEF position in meters
            vel (:obj:`numpy.ndarray`): triplet of floats representing ECEF velocity in meters / sec

        Returns:
            (tuple): tuple containing:
                * that (:obj:`numpy.ndarray`): Tangential unit vector.
                * chat (:obj:`numpy.ndarray`): Cross track unit vector.
                * nhat (:obj:`numpy.ndarray`): Normal unit vector pointing downwards.
        '''
        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef cartesian_t _t
        cdef cartesian_t _c
        cdef cartesian_t _n
        cdef int ii

        for ii in range(3):
            _pos[ii] = pos[ii]
            _vel[ii] = vel[ii]
        
        self.c_ellipsoid.TCNbasis(_pos,_vel,_t,_c,_n)

        that = np.empty(3, dtype=np.double)
        cdef double[:] thatview = that

        chat = np.empty(3, dtype=np.double)
        cdef double[:] chatview = chat

        nhat = np.empty(3, dtype=np.double)
        cdef double[:] nhatview = nhat

        for ii in range(3):
            thatview[ii] = _t[ii]
            chatview[ii] = _c[ii]
            nhatview[ii] = _n[ii]

        return (that, chat, nhat)

# end of file
