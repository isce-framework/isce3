import os

import numpy as np
from pytest import fixture
import numpy.testing as npt

import isce3
from nisar.products.readers import RSLC
import iscetest


class TestRSLCReader:
    """
    CLass that tests the RSLC reader.
    """

    @fixture
    def rslc_product(self):
        rslc_file = f'{iscetest.data}/envisat.h5'

        if not os.path.isfile(rslc_file):
            raise ValueError(
                f'RSLC file ("{rslc_file}") not found. If you have changed the'
                ' path of this file, please point `rslc_file` to the location'
                ' of the new RSLC reader test data.')

        return RSLC(hdf5file=rslc_file)

    def test_get_product_level(self, rslc_product):
        assert rslc_product.getProductLevel() == 'L1'

    def test_product_type(self, rslc_product):
        assert rslc_product.productType == 'RSLC'

    def test_get_noise_equivalent_backscatter_same_dimensions(self,
                                                              rslc_product):

        for frequency, pols in rslc_product.polarizations.items():

            for pol in pols:

                noise_product = rslc_product.getNoiseEquivalentBackscatter(
                    frequency=frequency, pol=pol)

                resampled_noise_product = \
                    rslc_product.getResampledNoiseEquivalentBackscatter(
                        sensing_times=noise_product.az_time)

                assert (noise_product.ref_epoch ==
                        resampled_noise_product.ref_epoch)

                err_msg = ('Noise power in linear units from the following'
                           ' methods do not match:'
                           ' getNoiseEquivalentBackscatter() and'
                           ' getResampledNoiseEquivalentBackscatter()')
                npt.assert_allclose(
                    noise_product.power_linear,
                    resampled_noise_product.power_linear,
                    err_msg=err_msg)

                err_msg = ('Noise product slant-range distance from the'
                           ' following methods do not match:'
                           ' getNoiseEquivalentBackscatter() and'
                           ' getResampledNoiseEquivalentBackscatter()')
                npt.assert_allclose(
                    noise_product.slant_range,
                    resampled_noise_product.slant_range)

                err_msg = ('Noise product azimuth time from the following'
                           ' methods do not match:'
                           ' getNoiseEquivalentBackscatter() and'
                           ' getResampledNoiseEquivalentBackscatter()')
                npt.assert_allclose(noise_product.az_time,
                                    resampled_noise_product.az_time)

                err_msg = ('Frequencies do not match:'
                           f' {noise_product.freq_band}'
                           ' (getNoiseEquivalentBackscatter) and'
                           f' {resampled_noise_product.freq_band}'
                           ' (getResampledNoiseEquivalentBackscatter)')
                npt.assert_equal(noise_product.freq_band,
                                 resampled_noise_product.freq_band)

                err_msg = ('Polarizations do not match:'
                           f' {noise_product.txrx_pol}'
                           ' (getNoiseEquivalentBackscatter) and'
                           f' {resampled_noise_product.txrx_pol}'
                           ' (getResampledNoiseEquivalentBackscatter)')
                npt.assert_equal(noise_product.txrx_pol,
                                 resampled_noise_product.txrx_pol,
                                 err_msg=err_msg)

    def test_get_noise_equivalent_backscatter_radar_grid(self, rslc_product):

        for frequency, pols in rslc_product.polarizations.items():

            rslc_radar_grid = rslc_product.getRadarGrid(frequency=frequency)

            for pol in pols:

                noise_product = rslc_product.getNoiseEquivalentBackscatter(
                    frequency=frequency, pol=pol)

                assert noise_product.ref_epoch == rslc_radar_grid.ref_epoch

                resampled_noise_product = \
                    rslc_product.getResampledNoiseEquivalentBackscatter(
                        sensing_times=rslc_radar_grid.sensing_times,
                        slant_ranges=rslc_radar_grid.slant_ranges)

                assert (noise_product.ref_epoch ==
                        resampled_noise_product.ref_epoch)

                err_msg = ('Noise power in linear units from the following'
                           ' methods do not match:'
                           ' getNoiseEquivalentBackscatter() and'
                           ' getResampledNoiseEquivalentBackscatter()'
                           ' using the RSLC radargrid for frequency'
                           f' {frequency}')
                npt.assert_allclose(
                    [noise_product.power_linear.max(),
                     noise_product.power_linear.min()],
                    [resampled_noise_product.power_linear.max(),
                     resampled_noise_product.power_linear.min()],
                    err_msg=err_msg)

                err_msg = ('Noise product slant-range distance from the'
                           ' following methods do not match:'
                           ' getNoiseEquivalentBackscatter() and'
                           ' getResampledNoiseEquivalentBackscatter()'
                           ' using the RSLC radargrid for frequency'
                           f' {frequency}')
                npt.assert_(
                    resampled_noise_product.slant_range[0] >=
                    noise_product.slant_range[0] and
                    resampled_noise_product.slant_range[-1] <=
                    noise_product.slant_range[-1],
                    msg=err_msg
                    )
                err_msg = ('Noise product azimuth time from the following'
                           ' methods do not match:'
                           ' getNoiseEquivalentBackscatter() and'
                           ' getResampledNoiseEquivalentBackscatter()'
                           ' using the RSLC radargrid for frequency'
                           f' {frequency}')
                npt.assert_(resampled_noise_product.az_time[0] >=
                            noise_product.az_time[0] and
                            resampled_noise_product.az_time[-1] <=
                            noise_product.az_time[-1],
                            msg=err_msg)

                err_msg = ('Frequencies do not match:'
                           f' {noise_product.freq_band}'
                           ' (getNoiseEquivalentBackscatter) and'
                           f' {resampled_noise_product.freq_band}'
                           ' (getResampledNoiseEquivalentBackscatter'
                           ' using the RSLC radargrid for frequency'
                           f' {frequency})')
                npt.assert_equal(noise_product.freq_band,
                                 resampled_noise_product.freq_band)

                err_msg = ('Polarizations do not match:'
                           f' {noise_product.txrx_pol}'
                           ' (getNoiseEquivalentBackscatter) and'
                           f' {resampled_noise_product.txrx_pol}'
                           ' (getResampledNoiseEquivalentBackscatter'
                           ' using the RSLC radargrid for frequency'
                           f' {frequency})')
                npt.assert_equal(noise_product.txrx_pol,
                                 resampled_noise_product.txrx_pol,
                                 err_msg=err_msg)
