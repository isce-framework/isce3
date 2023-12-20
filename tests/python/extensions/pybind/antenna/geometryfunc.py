#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt

from isce3.ext.isce3 import antenna as ant
from isce3.ext.isce3.core import Ellipsoid, Quaternion
from isce3.ext.isce3.geometry import DEMInterpolator as DEMInterp


class TestGeometryFunc:
    # Tolerances
    atol = 1e-2
    rtol = 1e-4

    # Input Parameters:
    # wavelength in (m)
    wl = 0.24
    # mechanical boresight angle,zero or positive, (deg)
    # if zero, means nadir looking antenna!
    mb_ang = 37.0
    # magnitude of spacecraft relative velocity (m/s)
    sc_vel = 7566.7
    # spacecraft position in LLH (deg,deg,m)
    sc_pos_llh = [-116.8531, 41.0549, 755451.5]
    # quaternion vector from spacecraft to ECEF
    quat_vec = [-0.17266, 0.71085, 0.56772, 0.37759]
    # mean dem height for all angles wrt WGS84 (m)
    dem_hgt = 200.0

    # Calculated Objects/Parameters:
    # EL and AZ angles in Antenna Frame all in (deg)
    el_deg = -mb_ang
    az_deg = 0.0
    # angles in radians
    el_rad = np.deg2rad(el_deg)
    az_rad = np.deg2rad(az_deg)
    # get WGS84  ellipsoid and quaternion objects
    wgs84 = Ellipsoid()
    q_ant2sc = Quaternion(*np.deg2rad([-90, mb_ang, 0.0]))
    q_sc2ecef = Quaternion(quat_vec)
    q_ant2ecef = q_sc2ecef * q_ant2sc
    # get spacecraft position and along-track velocity in ECEF
    sc_pos_llh_rad = sc_pos_llh[:]
    sc_pos_llh_rad[:2] = np.deg2rad(sc_pos_llh_rad[:2])
    sc_pos_ecef = wgs84.lon_lat_to_xyz(sc_pos_llh_rad)
    sc_vel_ecef = sc_vel * q_sc2ecef.rotate([1, 0, 0])
    # DEM interpolator object
    dem_interp = DEMInterp(dem_hgt)

    def test_ant2rgdop_scalar(self):
        slantrange, doppler, convergence = ant.ant2rgdop(
            self.el_rad, self.az_rad, self.sc_pos_ecef,
            self.sc_vel_ecef, self.q_ant2ecef, self.wl,
            self.dem_interp)

        npt.assert_allclose(slantrange,
                            self.sc_pos_llh[2] - self.dem_hgt,
                            rtol=self.rtol*0.5, atol=0.5,
                            err_msg="Wrong slantrange!")

        npt.assert_allclose(doppler, 0.0,
                            rtol=self.rtol, atol=self.atol,
                            err_msg="Wrong doppler!")

        npt.assert_equal(convergence, True,
                         err_msg="Wrong convergence flag!")

    def test_ant2rgdop_vector(self):
        el_vec = np.deg2rad([self.el_deg, self.el_deg + 0.1])
        slantrange, doppler, convergence = ant.ant2rgdop(
            el_vec, self.az_rad, self.sc_pos_ecef,
            self.sc_vel_ecef, self.q_ant2ecef, self.wl,
            self.dem_interp)

        npt.assert_allclose(slantrange[0],
                            self.sc_pos_llh[2] - self.dem_hgt,
                            rtol=self.rtol*0.5, atol=0.5,
                            err_msg="Wrong first slantrange!")

        npt.assert_equal(slantrange[1] > slantrange[0],
                         True, err_msg="Wrong second slantrange")

        npt.assert_allclose(doppler, 0.0,
                            rtol=self.rtol, atol=self.atol,
                            err_msg="Wrong doppler!")

        npt.assert_equal(convergence, True,
                         err_msg="Wrong convergence flag!")

    def test_ant2geo_scalar(self):
        llh, convergence = ant.ant2geo(
            self.el_rad, self.az_rad, self.sc_pos_ecef,
            self.q_ant2ecef, self.dem_interp)

        llh[:2] = np.rad2deg(llh[:2])

        npt.assert_allclose(llh[0], self.sc_pos_llh[0],
                            rtol=self.rtol, atol=self.atol,
                            err_msg="Wrong longitude!")

        npt.assert_allclose(llh[1], self.sc_pos_llh[1],
                            rtol=self.rtol, atol=self.atol,
                            err_msg="Wrong latitude!")

        npt.assert_allclose(llh[2], self.dem_hgt,
                            rtol=self.rtol*0.5, atol=0.5,
                            err_msg="Wrong dem height!")

        npt.assert_equal(convergence, True,
                         err_msg="Wrong convergence flag!")

    def test_ant2geo_vector(self):
        llh_list, convergence = ant.ant2geo(
            [self.el_rad, self.el_rad], self.az_rad,
            self.sc_pos_ecef, self.q_ant2ecef, self.dem_interp)

        for llh in llh_list:
            llh[:2] = np.rad2deg(llh[:2])

            npt.assert_allclose(llh[0], self.sc_pos_llh[0],
                                rtol=self.rtol, atol=self.atol,
                                err_msg="Wrong longitude!")

            npt.assert_allclose(llh[1], self.sc_pos_llh[1],
                                rtol=self.rtol, atol=self.atol,
                                err_msg="Wrong latitude!")

            npt.assert_allclose(llh[2], self.dem_hgt,
                                rtol=self.rtol*0.5, atol=0.5,
                                err_msg="Wrong dem height!")

        npt.assert_equal(convergence, True,
                         err_msg="Wrong convergence flag!")


def test_range_az_to_xyz():
    # Configuration for flying north at (lat, lon) = (0, 0):
    # Y-axis points north,  Z-axis (boresight) points down.
    # X=cross(Y,Z) must point west.
    q_ant2ecef = Quaternion(np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ]))
    # Looking along the equator (AZ=0), the Earth's radius of curvature is just
    # the semimajor axis.
    az = 0.0
    ellipsoid = Ellipsoid()
    a = ellipsoid.a
    # Height of platform and terrain.
    hp = 700e3
    ht = 0.0
    radar_xyz = np.array([a + hp, 0, 0])
    dem = DEMInterp(ht)
    # Pick some range > (hp - ht)
    r = 900e3
    # Expected solution from law of cosines (negative for left-looking)
    lon = np.arccos(((a + ht)**2 + (a + hp)**2 - r**2) /
                    (2 * (a + ht) * (a + hp)))
    expected_xyz = ellipsoid.lon_lat_to_xyz([-lon, 0, ht])
    # Since we're looking straight down let's search 0 <= EL <= 90 deg
    # for left-looking
    el_min, el_max = 0, np.pi/2

    xyz = ant.range_az_to_xyz(r, az, radar_xyz, q_ant2ecef, dem, el_min, el_max)

    npt.assert_allclose(xyz, expected_xyz)

    # Try another case with AZ != 0.  Don't have a simple closed-form expression
    # for the answer, but can check a few invariants.
    az = 0.1
    xyz = ant.range_az_to_xyz(r, az, radar_xyz, q_ant2ecef, dem, el_min, el_max)

    # Range is correct
    npt.assert_allclose(r, np.linalg.norm(xyz - radar_xyz))
    # +AZ points forward when left-looking
    npt.assert_(xyz[2] > 0)
