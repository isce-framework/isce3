import os

import numpy.testing as npt
import numpy as np

from nisar.noise import est_noise_power_from_raw
from nisar.noise.noise_estimation_from_raw import (
    TooShortNoiseRangeBlockWarning)
from nisar.products.readers.Raw import Raw
import iscetest


def pow2db(p: np.ndarray) -> np.ndarray:
    """Linear power to dB """
    return 10 * np.log10(p)


class TestNoiseEstFromRaw:
    # sub directory for all test files under "isce3/tests/data"
    sub_dir = 'bf'

    # sample single-pol single-band L0B file (noise-only)
    l0b_file = 'REE_L0B_ECHO_DATA_NOISE_EST.h5'

    # Absolute tolerance of max variation around mean noise power within
    # 240-km swath excluding the edges where the data is scaled down by
    # antenna-dependent DBF coeffs.
    # Note that even for constant noise power over all RX channels, when DBF
    # coefs are power normalized, there is some small variation of noise
    # power across swath due to expected ripples in 3-tap DBF envelope as
    # opposed to peak-normalized coeffs where there will be around +/-1 dB
    # induced variation of noise power.
    # This tolerance includes both DBF-related envelop variation and variance
    # of estimate itself which is the function of block size and the method.
    atol_db = 0.9

    # swath defined by slant range coverage (start, stop) in (km)
    # to be within approximately 240 km on the ground for NISAR.
    sw_km = (880., 1035.)

    # number of range blocks across entire swath
    n_rg_blk = 45

    # expected mean noise power (dB) over desired swath used for validation
    mean_db = 34.8

    # parser L0B
    raw = Raw(hdf5file=os.path.join(iscetest.data, sub_dir, l0b_file))

    # get the slant range slices for desired range blocks once
    # within desired swath from the default MVE noise power est.
    _ns_prod = est_noise_power_from_raw(raw, num_rng_block=n_rg_blk)[0]
    # find slant range slice of noise product for desired swath coverage
    _sr_km = _ns_prod.slant_range * 1e-3
    slice_sr = slice(*np.searchsorted(_sr_km, sw_km))
    print(f'Expected mean noise power within {sw_km} (km, km) '
          f'-> {mean_db:.3f} (dB)')

    def _validate_noise_product(self, ns_prod):
        p_db = pow2db(ns_prod[0].power_linear[self.slice_sr])
        npt.assert_allclose(p_db[~np.isnan(p_db)], self.mean_db,
                            atol=self.atol_db,
                            err_msg='Too larger noise power variation!')

    def test_mve_default(self):
        ns_prod = est_noise_power_from_raw(
            self.raw, num_rng_block=self.n_rg_blk, algorithm='MVE')
        self._validate_noise_product(ns_prod)

    def test_mve_mean(self):
        ns_prod = est_noise_power_from_raw(
            self.raw, num_rng_block=self.n_rg_blk, algorithm='MVE',
            diff_method='mean')
        self._validate_noise_product(ns_prod)

    def test_mve_diff(self):
        ns_prod = est_noise_power_from_raw(
            self.raw, num_rng_block=self.n_rg_blk, algorithm='MVE',
            diff_method='diff')
        self._validate_noise_product(ns_prod)

    def test_mee_default(self):
        ns_prod = est_noise_power_from_raw(
            self.raw, num_rng_block=self.n_rg_blk, algorithm='MEE')
        self._validate_noise_product(ns_prod)

    def test_mee_cpi2(self):
        ns_prod = est_noise_power_from_raw(
            self.raw, num_rng_block=self.n_rg_blk, algorithm='MEE', cpi=2)
        self._validate_noise_product(ns_prod)

    def test_mee_cpi4(self):
        ns_prod = est_noise_power_from_raw(
            self.raw, num_rng_block=self.n_rg_blk, algorithm='MEE', cpi=4)
        self._validate_noise_product(ns_prod)

    def test_too_short_noise_range_block_warn(self):
        with npt.assert_warns(TooShortNoiseRangeBlockWarning):
            est_noise_power_from_raw(
                self.raw, num_rng_block=1000, algorithm='MVE')
