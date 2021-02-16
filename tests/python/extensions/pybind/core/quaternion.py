#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt

import pybind_isce3 as isce3


class SetUp:
    """Set up const/vectors for testing"""
    # relative and absolute tolerances
    rtol = 1e-7
    atol = 1e-9

    # Euler Angles in radians
    ypr_ang = np.deg2rad([-0.9, 0.06, 0.15])

    # expected quaternion for the above YPR angles
    quat_ypr = [0.99996815847,
                0.00131306830238,
                0.00051330142620,
                -0.0078545784512]

    # 3-D Vectors , Rotation Matrix and Quaternion Vectors
    sc_pos = np.asarray(
        [-2434573.80388191110,
         -4820642.06528653484,
         4646722.94036952127])

    sc_vel = np.asarray(
        [522.99592536068,
         5107.80853161647,
         5558.15620986960])

    rot_mat = np.asarray(
        [[0.0, 0.99987663, -0.01570732],
         [-0.79863551, -0.0094529, -0.60174078],
         [-0.60181502, 0.01254442, 0.79853698]])

    # expected quaternions for rot_mat
    quat_rotmat = [0.66878324,
                   0.22962791, 
                   0.21909479,
                   -0.67230757]

    quat_ant2ecf = np.asarray(
        [0.14889715185,
         0.02930644114,
         -0.90605724862,
         -0.39500763650])

    # expected geocentric MB  angle per above ant2ecef 
    # quaternions and SC vel/pos in degrees 
    mb_ang_deg  = 37.0;   

    
class TestConstructors(SetUp):

    def test_default(self):
        q = isce3.core.Quaternion()
        npt.assert_allclose(q(), [1, 0, 0, 0], rtol=self.rtol,
                            atol=self.atol)

    def test_vector4(self):
        q = isce3.core.Quaternion(2.0 * self.quat_ant2ecf)
        npt.assert_allclose(q(), self.quat_ant2ecf,
                            rtol=self.rtol, atol=self.atol)

    def test_vector3(self):
        q = isce3.core.Quaternion(self.sc_vel)
        # normalize 3-D vector and insert zero at the start
        v4 = (self.sc_vel / np.linalg.norm(self.sc_vel)).tolist()
        v4.insert(0, 0)        
        npt.assert_allclose(q(), v4, rtol=self.rtol, atol=self.atol)

    def test_rotmat(self):
        q = isce3.core.Quaternion(self.rot_mat)
        npt.assert_allclose(q(), self.quat_rotmat, rtol=self.rtol,
                            atol=self.atol)

    def test_ypr(self):
        q = isce3.core.Quaternion(*self.ypr_ang)
        npt.assert_allclose(q(), self.quat_ypr, rtol=self.rtol,
                            atol=self.atol)

    def test_eulerangles(self):
        elr = isce3.core.EulerAngles(*self.ypr_ang)
        q = isce3.core.Quaternion(elr)
        npt.assert_allclose(q(), self.quat_ypr, rtol=self.rtol,
                            atol=self.atol)

    def test_quaternion(self):
        q1 = isce3.core.Quaternion(self.quat_ant2ecf)
        q2 = isce3.core.Quaternion(q1)
        npt.assert_allclose(q1(), q2(), rtol=self.rtol, atol=self.atol)

        

class TestMethods(SetUp):
    
    def test_to_ypr(self):        
        q = isce3.core.Quaternion(*self.ypr_ang)        
        npt.assert_allclose(q.to_ypr(), self.ypr_ang, rtol=self.rtol,
                            atol=self.atol)

    def test_to_euler_angles(self):
        q = isce3.core.Quaternion(*self.ypr_ang)
        elr = q.to_euler_angles()
        npt.assert_allclose([elr.yaw, elr.pitch, elr.roll], self.ypr_ang,
                            rtol=self.rtol, atol=self.atol)

    def test_rotate(self):
        uq_ant2ecf = isce3.core.Quaternion(self.quat_ant2ecf)
        ant_ecf = uq_ant2ecf.rotate([0, 0, 1])
        center_ecf =  -self.sc_pos / np.linalg.norm(self.sc_pos);
        est_mb_ang  = np.rad2deg(np.arccos(center_ecf.dot(ant_ecf)));
        npt.assert_allclose(est_mb_ang, self.mb_ang_deg, atol = 1e-1)


    def test_is_approx(self):
        q1 = isce3.core.Quaternion(*self.ypr_ang)
        q2 = isce3.core.Quaternion(*self.ypr_ang + 1e-4)
        npt.assert_(not q1.is_approx(q2))
        npt.assert_(q1.is_approx(q1))

    def test_conjugate(self):
        q = isce3.core.Quaternion(self.quat_ant2ecf)
        quat_conj = self.quat_ant2ecf
        quat_conj[1:] *= -1
        npt.assert_allclose(q.conjugate()(), quat_conj,
                            rtol=self.rtol, atol=self.atol)

        

class TestOperators(SetUp):

    def test_mul(self):
        q1 = isce3.core.Quaternion()
        q2 = isce3.core.Quaternion(*self.ypr_ang)
        q3 = q1 * q2
        npt.assert_allclose(q3(), q2(), rtol=self.rtol, atol=self.atol)

    def test_imul(self):
        q1 = isce3.core.Quaternion()
        q2 = isce3.core.Quaternion(*self.ypr_ang)
        q1 *= q2
        npt.assert_allclose(q1(), q2(), rtol=self.rtol, atol=self.atol)    

