#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from Serialization cimport load_archive
from Cartesian cimport cartesian_t
from Orbit cimport Orbit, orbitInterpMethod
import numpy as np
cimport numpy as np


cdef class pyOrbit:
    '''
    Python wrapper for isce::core::Orbit

    Note:
        Always set the number of state vectors or use addStateVector method before calling interpolation methods.

    Args:
        basis (Optional[int]: 0 for SCH, 1 for WGS84
        nVectors (Optional [int]: Number of state vectors
    '''
    cdef Orbit *c_orbit
    cdef bool __owner


    methods = { 'hermite': orbitInterpMethod.HERMITE_METHOD,
                'sch' :  orbitInterpMethod.SCH_METHOD,
                'legendre': orbitInterpMethod.LEGENDRE_METHOD}


    def __cinit__(self, basis=1, nVectors=0):
        '''
        Pre-constructor that creates a C++ isce::core::Orbit object and binds it to python instance.
        '''
        self.c_orbit = new Orbit(basis,nVectors)
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_orbit

    @staticmethod
    def bind(pyOrbit orb):
        new_orb = pyOrbit()
        del new_orb.c_orbit
        new_orb.c_orbit = orb.c_orbit
        new_orb.__owner = False
        return new_orb

    @property
    def basis(self):
        '''
        int: Basis code
        '''
        return self.c_orbit.basis

    @basis.setter
    def basis(self, int code):
        '''
        Set the basis code

        Args:
            a (int) : Value of basis code
        '''
        self.c_orbit.basis = code

    @property
    def nVectors(self):
        '''
        int: Number of state vectors.
        '''
        return self.c_orbit.nVectors


    @nVectors.setter
    def nVectors(self, int N):
        '''
        Set the number of state vectors.

        Args:
            N (int) : Number of state vectors.
        '''
        if (N < 0):
            raise ValueError('Number of state vectors cannot be < 0')

        self.c_orbit.nVectors = N
        self.c_orbit.UTCtime.resize(N)
        self.c_orbit.position.resize(3*N)
        self.c_orbit.velocity.resize(3*N)

    @property
    def UTCtime(self):
        '''
        Returns double precision time tags w.r.t reference epoch.

        Note:
            Returns actual reference to underlying C++ vector.

        Returns:
            numpy.array: double precision times corresponding to state vectors
        '''
        cdef int N = self.nVectors
        res = np.asarray(<double[:N]>(self.c_orbit.UTCtime.data()))
        return res

    @UTCtime.setter
    def UTCtime(self, times):
        '''
        Set the UTC times using a list or array.

        Args:
            times (list or np.array): UTC times corresponding to state vectors.
        '''
        if (self.nVectors != len(times)):
            raise ValueError("Invalid input size (expected list of length "+str(self.nVectors)+")")
        cdef int ii
        cdef int N = self.nVectors
        cdef double[::1] timearr = <double[:3*N]>(self.c_orbit.UTCtime.data())
        for ii in range(N):
            timearr[ii] = times[ii]

    @property
    def position(self):
        '''
        Note:
            Returns actual reference to underlying C++ vector.

        Returns:
            np.array[nx3]: Array of positions corresponding to state vectors.
        '''
        cdef int N = self.nVectors
        pos = np.asarray(<double[:N,:3]>(self.c_orbit.position.data()))
        return pos

    @position.setter
    def position(self, pos):
        '''
        Set the positions using a list or array.

        Args:
            pos (np.array[nx3]): Array of positions.
        '''
        if (self.nVectors != len(pos)):
            raise ValueError("Invalid input size (expected list of length/array "+str(self.nVectors))
        
        cdef int N = self.nVectors
        cdef int ii, jj

        for ii in range(N):
            row = pos[ii]
            for jj in range(3): 
                self.c_orbit.position[ii*3+jj] = row[jj]
    
    
    @property
    def velocity(self):
        '''
        Note:
            Returns actual reference to underlying C++ vector.

        Returns:
            np.array[nx3]: Array of velocities corresponding to state vectors.
        '''
        cdef int N = self.nVectors
        vel = np.asarray(<double[:N,:3]>(self.c_orbit.velocity.data()))
        return vel
    
    
    @velocity.setter
    def velocity(self, vel):
        '''
        Set the velocities using a list or array.

        Args:
            vel (np.array[nx3]): Array of velocities)
        '''

        if (self.nVectors != len(vel)):
            raise ValueError("Invalid input size (expected list/array of length "+str(self.nVectors)+")")
       
        cdef int N = self.nVectors
        cdef int ii, jj

        for ii in range(N):
            row = vel[ii]
            for jj in range(3):
                self.c_orbit.velocity[ii*3+jj] = row[jj]
    
    def copy(self, orb):
        '''
        Copy from a python object compatible with orbit.

        Args:
            orb (object): Any object with same attributes as pyOrbit

        Returns:
            None
        '''
        try:
            self.basis = orb.basis
            self.nVectors = orb.nVectors
            self.UTCtime = orb.UTCtime
            self.position = orb.position
            self.velocity = orb.velocity
        except:
            raise ValueError("Object passed in to copy is incompatible with object of type pyOrbit.")
    
    def dPrint(self):
        '''
        Debug print of underlying C++ structure.

        Returns:
            None
        '''
        self.printOrbit()

    def getPositionVelocity(self, double epoch):
        '''
        Interpolate the orbit at given epoch.

        Args:
            epoch (float): Floating point number representing UTC time

        Returns:
            tuple:
                * np.array[3] - Interpolated position at epoch
                * np.array[3] - Interpolated velocity at epoch
        '''
        cdef cartesian_t _pos
        cdef double[:] _posview = <double[:3]>(&_pos[0])

        cdef cartesian_t _vel
        cdef double[:] _velview = <double[:3]>(&_vel[0])

        self.c_orbit.getPositionVelocity(epoch,_pos,_vel)

        return (np.asarray(_posview.copy()), np.asarray(_velview.copy()))
    
    def getStateVector(self, int index):
        '''Return state vector based on index.

        Args:
            index (int): Integer between 0 and self.nVectors-1

        Returns:
            tuple:
                float - UTCtime of state vector
                np.array[3] - position
                np.array[3] - velocity
        '''
       
        cdef cartesian_t pos
        cdef double[:] posview = <double[:3]>(&pos[0])

        cdef cartesian_t vel
        cdef double[:] velview = <double[:3]>(&vel[0])

        cdef double epoch

        self.c_orbit.getStateVector(index, epoch, pos, vel)

        return (epoch, np.asarray(posview.copy()), np.asarray(velview.copy()))

    def setStateVector(self, int index, double epoch, pos, vel):
        '''
        Set state vector with given epoch, position and velocity.

        Args:
           index (int) : Index to set
           epoch (float) : Epoch in UTC time
           pos (np.array[3]) : Position
           vel (np.array[3]) : Velocity

        Returns:
            None
        '''

        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef int ii
        for ii in range(3):
            _pos[ii] = pos[ii]

        for ii in range(3):
            _vel[ii] = vel[ii]

        self.c_orbit.setStateVector(index,epoch,_pos,_vel)
    
    def addStateVector(self, double epoch, pos, vel):
        '''
        Add a state vector. The index for insertion into C++ data structure is automatically determined using the epoch information.

        Args:
            epoch (float): Epoch in UTC time
            pos (np.array[3]): Position
            vel (np.array[3]): Velocity

        Returns:
            None
        '''
        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef int ii
        for ii in range(3):
            _pos[ii] = pos[ii]

        for ii in range(3):
            _vel[ii] = vel[ii]
        
        self.c_orbit.addStateVector(epoch, _pos, _vel)
    
    def interpolate(self, epoch, method='hermite'):
        '''
        Interpolate orbit at a given epoch.

        Args:
            epoch (float or np.array[N]): Epoch in UTC time
            method (Optional[str]): hermite, sch or legendre 

        Returns:
            tuple:
                * float or np.array[N]: Status flag. 0 for success.
                * np.array[3] or np.array[Nx3]: Interpolated position
                * np.array[3] or np.array[Nx3]: Interpolated velocity
        '''

        cdef orbitInterpMethod alg = self.methods[method]

        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef int ret
        
        if np.isscalar(epoch): 
            ret = self.c_orbit.interpolate(epoch,_pos,_vel,alg)
            return (ret, np.asarray((<double[:3]>(&(_pos[0]))).copy()), np.asarray((<double[:3]>(&(_vel[0]))).copy()))

        epoch = np.atleast_1d(epoch)
        cdef int Npts = epoch.shape[0]
        cdef int ii, jj
        flag = np.empty((Npts), dtype=np.int)
        cdef long[:] flagview = flag

        pos = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] posview = pos

        vel = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] velview = vel
        
        for ii in range(Npts):
            ret = self.c_orbit.interpolate(epoch[ii], _pos, _vel, alg)
            flagview[ii] = ret

            for jj in range(3):
                posview[ii,jj] = _pos[jj]
                
            for jj in range(3):
                velview[ii,jj] = _vel[jj]

        return (flag, pos, vel)

    
    def interpolateWGS84Orbit(self, epoch):
        '''
        Interpolate orbit at given epoch using WGS84 interpolation method.

        Args:
            epoch (float or np.array[N]): Epoch in UTC time
            method (Optional[str]): hermite, sch or legendre 

        Returns:
            tuple:
                * float or np.array[N]: Status flag. 0 for success.
                * np.array[3] or np.array[Nx3]: Interpolated position
                * np.array[3] or np.array[Nx3]: Interpolated velocity
        '''
        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef int ret
        
        if np.isscalar(epoch): 
            ret = self.c_orbit.interpolateWGS84Orbit(epoch,_pos,_vel)
            return (ret, np.asarray((<double[:3]>(&(_pos[0]))).copy()), np.asarray((<double[:3]>(&(_vel[0]))).copy()))

        epoch = np.atleast_1d(epoch)
        cdef int Npts = epoch.shape[0]
        cdef int ii, jj
        flag = np.empty((Npts), dtype=np.int)
        cdef long[:] flagview = flag

        pos = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] posview = pos

        vel = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] velview = vel
        
        for ii in range(Npts):
            ret = self.c_orbit.interpolateWGS84Orbit(epoch[ii], _pos, _vel)
            flagview[ii] = ret

            for jj in range(3):
                posview[ii,jj] = _pos[jj]
                
            for jj in range(3):
                velview[ii,jj] = _vel[jj]

        return (flag, pos, vel)

    def interpolateSCHOrbit(self, epoch):
        '''
        Interpolate orbit at given epoch using SCH interpolation method.

        Args:
            epoch (float or np.array[N]): Epoch in UTC time
            method (Optional[str]): hermite, sch or legendre 

        Returns:
            tuple:
                * float or np.array[N]: Status flag. 0 for success.
                * np.array[3] or np.array[Nx3]: Interpolated position
                * np.array[3] or np.array[Nx3]: Interpolated velocity
        '''
        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef int ret
        
        if np.isscalar(epoch): 
            ret = self.c_orbit.interpolateSCHOrbit(epoch,_pos,_vel)
            return (ret, np.asarray((<double[:3]>(&(_pos[0]))).copy()), np.asarray((<double[:3]>(&(_vel[0]))).copy()))

        epoch = np.atleast_1d(epoch)
        cdef int Npts = epoch.shape[0]
        cdef int ii, jj
        flag = np.empty((Npts), dtype=np.int)
        cdef long[:] flagview = flag

        pos = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] posview = pos

        vel = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] velview = vel
        
        for ii in range(Npts):
            ret = self.c_orbit.interpolateSCHOrbit(epoch[ii], _pos, _vel)
            flagview[ii] = ret

            for jj in range(3):
                posview[ii,jj] = _pos[jj]
                
            for jj in range(3):
                velview[ii,jj] = _vel[jj]

        return (flag, pos, vel)    

    def interpolateLegendreOrbit(self, epoch):
        '''
        Interpolate orbit at given epoch using Legendre interpolation method.

        Args:
            epoch (float or np.array[N]): Epoch in UTC time
            method (Optional[str]): hermite, sch or legendre 

        Returns:
            tuple:
                * int or np.array[N]: Status flag. 0 for success.
                * np.array[3] or np.array[Nx3]: Interpolated position
                * np.array[3] or np.array[Nx3]: Interpolated velocity
        '''
        cdef cartesian_t _pos
        cdef cartesian_t _vel
        cdef int ret
        
        if np.isscalar(epoch): 
            ret = self.c_orbit.interpolateLegendreOrbit(epoch,_pos,_vel)
            return (ret, np.asarray((<double[:3]>(&(_pos[0]))).copy()), np.asarray((<double[:3]>(&(_vel[0]))).copy()))

        epoch = np.atleast_1d(epoch)
        cdef int Npts = epoch.shape[0]
        cdef int ii, jj
        flag = np.empty((Npts), dtype=np.int)
        cdef long[:] flagview = flag

        pos = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] posview = pos

        vel = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] velview = vel
        
        for ii in range(Npts):
            ret = self.c_orbit.interpolateLegendreOrbit(epoch[ii], _pos, _vel)
            flagview[ii] = ret

            for jj in range(3):
                posview[ii,jj] = _pos[jj]
                
            for jj in range(3):
                velview[ii,jj] = _vel[jj]

        return (flag, pos, vel)

    
    def computeAcceleration(self, epoch):
        '''
        Compute accelerations at given epoch.

        Args:
            epoch (float or np.array[N]): Epoch in UTC time

        Returns:
            tuple:
                * int or np.array[N]: Status flag. 0 for success.
                * np.array[3] or np.array[Nx3]: 3D acceleration
        '''
        cdef cartesian_t _acc
        cdef int ret

        if np.isscalar(epoch):
            ret = self.c_orbit.computeAcceleration(epoch,_acc)
            return (ret, np.asarray((<double[:3]>(&(_acc[0]))).copy()))

        epoch = np.atleast_1d(epoch)
        cdef int Npts = epoch.shape[0]
        cdef int ii, jj
        
        flag = np.empty((Npts), dtype=np.int)
        cdef long[:] flagview = flag

        acc = np.empty((Npts,3), dtype=np.double)
        cdef double[:,:] accview = acc

        for ii in range(Npts):
            ret = self.c_orbit.computeAcceleration(epoch[ii], _acc) 
            flagview[ii] = ret

            for jj in range(3):
                accview[ii,jj] = _acc[jj]

        return (flag, acc)
    
    def printOrbit(self):
        '''
        Debug printing of underlying C++ structure.

        Returns:
            None
        '''
        self.c_orbit.printOrbit()
    
    def loadFromHDR(self, filename, int basis=1):
        '''
        Load Orbit from a text file.

        Args:
            filename (str): Filename with state vectors
            basis (Optional[int]): Basis for state vectors

        Returns:
            None
        '''
        cdef bytes fname = pyStringToBytes(filename)
        cdef char *cstring = fname
        self.c_orbit.loadFromHDR(cstring, basis)
    
    def dumpToHDR(self, filename):
        '''
        Write orbit to a text file.

        Args:
            filename (str): Output file name

        Returns:
            None
        '''
        cdef bytes fname = pyStringToBytes(filename)
        cdef char *cstring = fname
        self.c_orbit.dumpToHDR(cstring)

    def archive(self, metadata):
        load_archive[Orbit](pyStringToBytes(metadata),
                            'Orbit',
                            self.c_orbit)

