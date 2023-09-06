import os
import numpy as np
import numpy.testing as npt

from isce3.antenna import geo2ant, rdr2ant
from isce3.geometry import DEMInterpolator, rdr2geo
from nisar.products.readers.SLC import SLC
from isce3.core.types import ComplexFloat16Decoder
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
    dset = ComplexFloat16Decoder(slc.getSlcDataset(freq_band, pol_txrx))
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
