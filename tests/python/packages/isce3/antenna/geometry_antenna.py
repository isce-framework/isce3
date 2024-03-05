import os
import numpy as np
import numpy.testing as npt

from isce3.antenna import (geo2ant, rdr2ant, sphere_range_az_to_xyz,
    get_approx_el_bounds, range_az_to_xyz)
from isce3.core import Ellipsoid, Quaternion
from isce3.geometry import DEMInterpolator, rdr2geo
from nisar.products.readers.SLC import SLC
import iscetest


class TestGeometryAntenna:

    # list of inputs

    # absolute tolerance for EL/AZ angles in deg
    atol_deg = 1e-3

    # SLC file with single point target
    file_slc = 'REE_RSLC_out17.h5'
    pol_txrx = 'HH'
    freq_band = 'A'

    # Target actual location in antenna frame copied from REE input config/RDF
    # file "REE_L0B_out17.rdf", that is EL (deg), AZ (deg), Height (m).
    # This will be used as reference for validation.
    tg_el = 0.0
    tg_az = 0.0
    tg_hgt = 0.0

    # parse slc
    slc = SLC(hdf5file=os.path.join(iscetest.data, file_slc))

    # parse dataset and find the (line, pixel) of the point target
    dset = slc.getSlcDatasetAsNativeComplex(freq_band, pol_txrx)
    tg_line, tg_pixel = np.unravel_index(abs(dset[:]).argmax(), dset.shape)

    # get radar grid
    rdr_grid = slc.getRadarGrid(freq_band)
    wavelength = rdr_grid.wavelength
    ant_side = rdr_grid.lookside

    # get slant range (m) and azimuth time (sec) of the point target
    # at peak location
    tg_sr = rdr_grid.slant_range(tg_pixel)
    tg_azt = rdr_grid.az_time_interval * tg_line + rdr_grid.sensing_start

    # get orbit and attitude
    orbit = slc.getOrbit()
    attitude = slc.getAttitude()

    # build DEMInterp from target height
    dem = DEMInterpolator(tg_hgt)

    def _validate(self, est_el_az):
        npt.assert_allclose(np.rad2deg(est_el_az), (self.tg_el, self.tg_az),
                            err_msg='Wrong (EL, AZ)!', atol=self.atol_deg)

    def test_rdr2ant(self):
        el_az = rdr2ant(self.tg_azt, self.tg_sr, self.orbit, self.attitude,
                        self.ant_side, self.wavelength, dem=self.dem)
        self._validate(el_az)

    def test_geo2ant(self):
        # get target llh
        tg_llh = rdr2geo(self.tg_azt, self.tg_sr, self.orbit, self.ant_side,
                         doppler=0, wavelength=self.wavelength, dem=self.dem)
        # get S/C pos and quaternions at target azimuth time
        pos, _ = self.orbit.interpolate(self.tg_azt)
        quat = self.attitude.interpolate(self.tg_azt)
        el_az = geo2ant(tg_llh, pos, quat)
        self._validate(el_az)


# Copied from unit test for rangeAzToXYZ
def test_sphere_range_az_to_xyz():
    # Configuration for flying north at (lat, lon) = (0, 0):
    # Y-axis points north,  Z-axis (boresight) points down.
    # X=cross(Y,Z) must point west.
    # Also rotate boresight 20 deg off nadir to imply a look side.
    q_ant2ecef = Quaternion(np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ])) * Quaternion(np.deg2rad(20), [0, 1, 0])
    # Looking along the equator (AZ=0), the Earth's radius of curvature is just
    # the semimajor axis.
    az = 0.0
    ellipsoid = Ellipsoid()
    a = ellipsoid.a
    # Height of platform and terrain.
    hp = 700e3
    ht = 0.0
    radar_xyz = np.array([a + hp, 0, 0])
    dem = DEMInterpolator(ht)
    # Pick some range > (hp - ht)
    r = 900e3
    # Expected solution from law of cosines (negative for left-looking)
    lon = np.arccos(((a + ht)**2 + (a + hp)**2 - r**2) /
                    (2 * (a + ht) * (a + hp)))
    expected_xyz = ellipsoid.lon_lat_to_xyz([-lon, 0, ht])

    xyz = sphere_range_az_to_xyz(r, az, radar_xyz, q_ant2ecef, a)

    npt.assert_allclose(xyz, expected_xyz)

    # Try another case with AZ != 0.  Don't have a simple closed-form expression
    # for the answer, but can check a few invariants.
    az = 0.1
    xyz = sphere_range_az_to_xyz(r, az, radar_xyz, q_ant2ecef, a)

    # Range is correct
    npt.assert_allclose(r, np.linalg.norm(xyz - radar_xyz))
    # +AZ points forward when left-looking
    npt.assert_(xyz[2] > 0)


def test_el_bounds():
    # These values come from an ALOS test case (ALPSRP144730690-L1.0) where
    # the boresight is 21.5 deg off nadir and the default EL search bounds of
    # [-45, 45] degrees are invalid because both ends of that arc are above the
    # surface of the Earth.
    r = 743588.0
    az = 0.0
    radar_xyz = np.array([-3015237.24427643, -5058067.5932234, 3913125.89004813])
    rcs2xyz = np.array([
        [ 0.68178977, -0.08206779,  0.72693025],
        [-0.6264125 , -0.57873936,  0.52217634],
        [ 0.37784929, -0.81137268, -0.44598687]])
    q = Quaternion(rcs2xyz)
    dem = DEMInterpolator()

    el_min, el_max = get_approx_el_bounds(r, az, radar_xyz, q, dem)

    # verify we have a smaller interval that doesn't cross nadir
    npt.assert_(np.rad2deg(el_min) > -21.5)
    npt.assert_(np.rad2deg(el_max) < 45.0)

    # make sure these EL bonds are valid for range_az_to_xyz (it doesn't crash)
    range_az_to_xyz(r, az, radar_xyz, q, dem, el_min=el_min, el_max=el_max)
