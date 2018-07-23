#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp cimport bool
from Pegtrans cimport Pegtrans


cdef class pyPegtrans:
    cdef Pegtrans *c_pegtrans
    cdef bool __owner

    def __cinit__(self):
        self.c_pegtrans = new Pegtrans()
        self.__owner = True
    def __dealloc__(self):
        if self.__owner:
            del self.c_pegtrans
    @staticmethod
    def bind(pyPegtrans pgt):
        new_pgt = pyPegtrans()
        del new_pgt.c_pegtrans
        new_pgt.c_pegtrans = pgt.c_pegtrans
        new_pgt.__owner = False
        return new_pgt

    @property
    def mat(self):
        a = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        for i in range(3):
            for j in range(3):
                a[i][j] = self.c_pegtrans.mat[i][j]
        return a
    @mat.setter
    def mat(self, a):
        if ((len(a) != 3) or (len(a[0]) != 3)):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            for j in range(3):
                self.c_pegtrans.mat[i][j] = a[i][j]
    @property
    def matinv(self):
        a = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        for i in range(3):
            for j in range(3):
                a[i][j] = self.c_pegtrans.matinv[i][j]
        return a
    @matinv.setter
    def matinv(self, a):
        if ((len(a) != 3) or (len(a[0]) != 3)):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            for j in range(3):
                self.c_pegtrans.matinv[i][j] = a[i][j]

    @property
    def ov(self):
        a = [0.,0.,0.]
        for i in range(3):
            a[i] = self.c_pegtrans.ov[i]
        return a
    @ov.setter
    def ov(self, a):
        if (len(a) != 3):
            print("Error: Invalid input size.")
            return
        for i in range(3):
            self.c_pegtrans.ov[i] = a[i]

    @property
    def radcur(self):
        return self.c_pegtrans.radcur
    @radcur.setter
    def radcur(self, double a):
        self.c_pegtrans.radcur = a

    def dPrint(self):
        print("Mat = "+str(self.mat)+", matinv = "+str(self.matinv)+", ov = "+str(self.ov)+
              ", radcur = "+str(self.radcur))

    def copy(self, pt):
        try:
            self.mat = pt.mat
            self.matinv = pt.matinv
            self.ov = pt.ov
            self.radcur = pt.radcur
        except:
            print("Error: Object passed in is incompatible with object of type pyPegtrans.")

    def radarToXYZ(self, pyEllipsoid a, pyPeg b):
        self.c_pegtrans.radarToXYZ(deref(a.c_ellipsoid),deref(b.c_peg))

    def convertXYZtoSCH(self, xyz):
        """
        Transform ECEF xyz coordinates to SCH.

        Args:
            xyz (np.array[3 or nx3]): triplet of floats representing XYZ.

        Returns:
            np.array[3 or nx3]: triplet of floats representing SCH.
        """
        # Standardize input coordinates
        xyz = np.atleast_2d(xyz)
        cdef unsigned long npts = xyz.shape[0]

        # Create output array and memory view
        sch = np.empty((npts, 3), dtype=np.double)
        cdef double[:,:] schview = sch

        # Loop over points
        cdef unsigned long i, j
        cdef cartesian_t xyz_in, sch_out
        for i in range(npts):
            # Copy current coordinate to cartesian_t
            for j in range(3):
                xyz_in[j] = xyz[i,j]
            # Perform conversion
            self.c_pegtrans.convertXYZtoSCH(xyz_in, sch_out)
            # Save result
            for j in range(3):
                schview[i,j] = sch_out[j]

        return np.squeeze(sch)

    def convertSCHtoXYZ(self, sch):
        """
        Transform SCH to ECEF xyz coordinates.

        Args:
            sch (np.array[3 or nx3]): triplet of floats representing SCH.

        Returns:
            np.array[3 or nx3]: triplet of floats representing ECEF XYZ.
        """
        # Standardize input coordinates
        sch = np.atleast_2d(sch)
        cdef unsigned long npts = sch.shape[0]

        # Create output array and memory view
        xyz = np.empty((npts, 3), dtype=np.double)
        cdef double[:,:] xyzview = xyz

        # Loop over points
        cdef unsigned long i, j
        cdef cartesian_t sch_in, xyz_out
        for i in range(npts):
            # Copy current coordinate to cartesian_t
            for j in range(3):
                sch_in[j] = sch[i,j]
            # Perform conversion
            self.c_pegtrans.convertSCHtoXYZ(sch_in, xyz_out)
            # Save result
            for j in range(3):
                xyzview[i,j] = xyz_out[j]

        return np.squeeze(xyz)

    def convertXYZdottoSCHdot(self, sch, xyzdot):
        """
        Transform ECEF xyz velocities to SCH velocities.

        Args:
            sch (np.array[3 or nx3]): triplet of floats representing SCH.
            xyzdot (np.array[3 or nx3]): triplet of floats representing XYZ velocities.

        Returns:
            np.array[3 or nx3]: triplet of floats representing SCH velocities.
        """
        # Standardize input coordinates
        sch = np.atleast_2d(sch)
        xyzdot = np.atleast_2d(xyzdot)
        cdef unsigned long npts = sch.shape[0]
        assert sch.shape == xyzdot.shape

        # Create output array and memory view
        schdot = np.empty((npts, 3), dtype=np.double)
        cdef double[:,:] schdotview = schdot

        # Loop over points
        cdef unsigned long i, j
        cdef cartesian_t sch_in, xyzdot_in, schdot_out
        for i in range(npts):
            # Copy current coordinates to cartesian_t
            for j in range(3):
                sch_in[j] = sch[i,j]
                xyzdot_in[j] = xyzdot[i,j]
            # Perform conversion
            self.c_pegtrans.convertXYZdotToSCHdot(sch_in, xyzdot_in, schdot_out)
            # Save result
            for j in range(3):
                schdotview[i,j] = schdot_out[j]

        return np.squeeze(schdot)

    def convertSCHdottoXYZdot(self, sch, schdot):
        """
        Transform SCH velocities to ECEF xyz velocities.

        Args:
            sch (np.array[3 or nx3]): triplet of floats representing SCH.
            schdot (np.array[3 or nx3]): triplet of floats representing SCH velocities.

        Returns:
            np.array[3 or nx3]: triplet of floats representing ECEF xyz velocities.
        """
        # Standardize input coordinates
        sch = np.atleast_2d(sch)
        schdot = np.atleast_2d(schdot)
        cdef unsigned long npts = sch.shape[0]
        assert sch.shape == schdot.shape

        # Create output array and memory view
        xyzdot = np.empty((npts, 3), dtype=np.double)
        cdef double[:,:] xyzdotview = xyzdot

        # Loop over points
        cdef unsigned long i, j
        cdef cartesian_t sch_in, schdot_in, xyzdot_out
        for i in range(npts):
            # Copy current coordinates to cartesian_t
            for j in range(3):
                sch_in[j] = sch[i,j]
                schdot_in[j] = schdot[i,j]
            # Perform conversion
            self.c_pegtrans.convertSCHdotToXYZdot(sch_in, schdot_in, xyzdot_out)
            # Save result
            for j in range(3):
                xyzdotview[i,j] = xyzdot_out[j]

        return np.squeeze(xyzdot)

    def SCHbasis(self, sch):
        """
        Computes transformation matrix to-and-from SCH and XYZ frames.

        Args:
            sch(np.array[3]): triplet of floats representing current SCH position.
        
        Returns:
            R_xyz2sch(np.array[3,3]): transformation array XYZ->SCH
            R_sch2xyz(np.array[3,3]): transformation array SCH->XYZ
        """
        sch = np.array(sch)

        # Allocate arrays for transformation arrays
        R_sch2xyz = np.zeros((3,3))
        R_xyz2sch = np.zeros((3,3))

        # Initialize cartesian objects
        cdef cartesian_t sch_in
        cdef cartmat_t M_sch2xyz
        cdef cartmat_t M_xyz2sch
        cdef i, j
        for j in range(3):
            sch_in[j] = sch[j]

        # Compute matrices
        self.c_pegtrans.SCHbasis(sch_in, M_xyz2sch, M_sch2xyz)

        # Copy data to outputs
        for i in range(3):
            for j in range(3):
                R_sch2xyz[i,j] = M_sch2xyz[i][j]
                R_xyz2sch[i,j] = M_xyz2sch[i][j]

        return R_xyz2sch, R_sch2xyz

# end of file
