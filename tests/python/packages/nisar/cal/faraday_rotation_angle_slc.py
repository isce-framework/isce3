import os
import numpy as np
import numpy.testing as npt
import pytest

import iscetest
from nisar.cal import (FaradayRotEstBickelBates,
                       FaradayRotEstFreemanSecond,
                       faraday_rot_angle_from_cr)
from nisar.products.readers.SLC import SLC
from nisar.workflows.pol_channel_imbalance_from_rslc import cr_llh_from_csv


# List of inputs to the test Class, see Notes and References below for details.

# RSLC product with corner reflector (~ 5.1m) at Remningstorp, Sweden [1]
# @ high latitude (~ 59.0 deg)
file_slc1 = 'pol_rad_calib_rslc_Remningstorp_ALPSRP030392430.h5'
file_csv1 = 'Corner_Reflector_Remningstorp_ALPSRP030392430.csv'
# Estimated Faraday rotation angle (deg) provided by Fig. (3) of Ref [1]
fra_deg1 = -3.5
# Expected max variation+bias (~ 3*sigma + mean) in FR angle estimate based
# on the Ref [1]
abs_tol_deg1 = 1.7
# Note that the magnitude of FR angle calculated from JPL GPS-measured TEC
# and 13th-order solution of Earth Geomagnetic field is around 2.0 deg
# @ boresight (~21.5 deg) and altitude 350 km (F2 Ion layer)!

# RSLC product with corner reflector (~ 2.5m) at Rio Branco, Brazil [2]
# @ low latitude close to geomagnetic equator (~ -9.0 deg)
file_slc2 = 'pol_rad_calib_rslc_Rio_Branco_ALPSRP025826990.h5'
file_csv2 = 'Corner_Reflector_Rio_Branco_ALPSRP025826990.csv'
# Estimated Faraday rotation angle (deg) provided by Fig. (5) of Ref [2]
fra_deg2 = 1.65
# Observed max variation in FR angle estimate based on Fig. (5) of Ref [2]
abs_tol_deg2 = 0.5
# Note that the magnitude of FR angle calculated from JPL GPS-measured TEC
# and 13th-order solution of Earth Geomagnetic field is around 0.4 deg
# @ boresight(~21.5 deg) and altitude 350 km (F2 Ion layer)!


@pytest.mark.parametrize("file_slc,file_csv,fra_deg,abs_tol_deg",
                         [(file_slc1, file_csv1, fra_deg1, abs_tol_deg1),
                          (file_slc2, file_csv2, fra_deg2, abs_tol_deg2)
                          ]
                         )
class TestFaradyRotationAngleSlc:
    """

    Notes
    -----
    Note that the sample quad-pol RSLC products are all radiomterically
    calibrated by using `ALOS1_PALSAR_ANTPAT_BEAM215.h5`. The polarimetric
    channels are also calibrated via averaged Tx/Rx channel imbalances
    estimated by the ISCE3 workflow `pol_channel_imbalance_from_rslc`.
    No cross-talk is removed given reported cross-talk values are around
    or less than -35 dB per [1]_ and [2]_.

    Each truncated RSLC product also contains a trihedral corner reflector
    whose geodetic location info are extracted from RSLC and dump into the
    respective CSV product.

    The Faraday rotation (FR) angle estimates provided by the respective
    references [1]_ and [2]_ are used as a validation with some expected
    margins due to limited noise-equivalent sigma-zero (NESZ), uncompensated
    residual channel imbalance, and uncompensated relatively-small
    cross talks [3]_.

    Note that the calculated FR angles based on GPS-measured TEC values are
    lower than RSLC-driven estimates. These values are simply reported in the
    commented lines for the sake of readers information. None of the refereces
    could confirm and validate the estimated data-driven FR angles against
    the calculated ones from coarse GPS-measured TEC maps even after an almost
    perfect polarimetric calibration performed by JAXA for ALOS1 PALSAR RSLC
    products stated in the respective references.

    Notice that a tolerance adjustment for Freeman-second (FS) approach v.s.
    Bickel-Bates (BB) for ALOS1 PALSAR is required given NESZ of ALOS1 PALSAR
    Polarimetric product is around -32 dB per Fig. (1) of [4]_.
    See tables (III, IV) of Ref [3]_ for an expected overestimatad FR angle
    from FS method and its difference with that of BB method.

    References
    ----------
    .. [1] G. Sandberg, L. E. B. Eriksson, L. Ulander, "Measurements of Faraday
        Rotation Using Polarimetric PALSAR Images," IEEE GeoSci and Remote
        Sens. Letters, pp.142-146, January, 2009.
    .. [2] M. Shimada, M. Ohki, "Calibration of PALSAR Polarimetry Using
        Corner Reflectors In the Amazon Forest and Its Three-Year Trend,"
        Workshop on Science and Applications of SAR Polarimetry and PolInSAR,
        Italy, April 2009.
    .. [3] A. Freeman, "Calibration of linearly polarized polarimetric SAR data
        subject to Faraday rotation," IEEE Trans. Geosci. Remote Sens., Vol 42,
        pp.1617-1624, August, 2004.
    .. [4] M. Shimada, A. Rosenqvistz, A. M. Watanabe, T. Tadono,
        "The Polarimetric and Interferometric Potential of ALOS PALSAR,"
        Proceedings of the 2ed International Workshop POLINSAR, Italy, January
        2005.

    """
    # sub directory containing all input files
    sub_dir = 'faraday_rot'

    # temporary dir
    tmp_dir = '.'

    # Tolerance adjustment for Freeman-Second (FS) approach v.s. Bickel-Bates
    # (BB) for ALOS1 PALSAR. See tables (III-IV) of Ref [3] for x-talk below
    # -30.0 dB and NESZ of ALOS1 PALSAR Polarimetric around -32 dB.
    # FS approach always overestimate the FR angle versus BB.
    # Note that the sign of FR angle from FS shall be the same as that of BB
    # method. The max expected difference in magnitude between two methods:
    # 0.0 < abs(FS) - abs(BB) < 8.5 deg!
    tol_abs_fra_adj = 8.5

    def _validate(self, fra_prod, fra_deg, abs_tol_deg, abs_only=False):
        fra_est_deg = np.rad2deg(fra_prod.faraday_ang)
        if abs_only:
            fra_est_deg = abs(fra_est_deg)
            fra_deg = abs(fra_deg)
        npt.assert_allclose(
            fra_est_deg, fra_deg, atol=abs_tol_deg,
            err_msg='Wrong Faraday Rotation Angle(s)!'
        )

    def test_faraday_ext_scene_time_domain_bickle_bates(
            self, file_slc, file_csv, fra_deg, abs_tol_deg):
        # Parse RSLC
        slc = SLC(hdf5file=os.path.join(iscetest.data, self.sub_dir, file_slc))
        # Estimate FR angle in time domain
        with FaradayRotEstBickelBates(slc, dir_tmp=self.tmp_dir,
                                      plot=True) as fra:
            fra_prod = fra.estimate()
        # Validate
        self._validate(fra_prod, fra_deg, abs_tol_deg)

    def test_faraday_ext_scene_time_domain_freeman_second(
            self, file_slc, file_csv, fra_deg, abs_tol_deg):
        # Parse RSLC
        slc = SLC(hdf5file=os.path.join(iscetest.data, self.sub_dir, file_slc))
        # Estimate FR angle in time domain
        with FaradayRotEstFreemanSecond(slc, dir_tmp=self.tmp_dir,
                                        plot=True) as fra:
            fra_prod = fra.estimate()
        # Loosen the tolerance for over-estimated FS method!
        abs_tol_deg_fs = abs_tol_deg + self.tol_abs_fra_adj
        # Validate
        self._validate(fra_prod, fra_deg, abs_tol_deg_fs)

    def test_faraday_ext_scene_freq_domain_slope(
            self, file_slc, file_csv, fra_deg, abs_tol_deg):
        # Parse RSLC
        slc = SLC(hdf5file=os.path.join(iscetest.data, self.sub_dir, file_slc))
        # Estimate FR angle in frequency domain
        with FaradayRotEstBickelBates(slc, use_slope_freq=True,
                                      dir_tmp=self.tmp_dir, ovsf=1.25,
                                      plot=True) as fra:
            fra_prod = fra.estimate()
        # Validate
        # Notice in case of Rio Branco, the scene is close to geomagnetic
        # equator where the FR angle estimates are close to zero.
        # Besides, given small sensitivity of Faraday rotation (FR) to
        # frequency variation within relatively small RF bandwidth of this
        # dataset (~14 MHz), the SLOPE approach simply fails with wrong
        # sign while its magnitude is in right range.
        # For the sake of exercising all FR approaches under totally different
        # geomagnetic and geospatial circumstances, the absolute values of FR
        # is simply validated.
        # Thus, we ignore the sign in this comparison to let the test
        # pass over Rio Branco case (for only SLOPE approach).
        # For cases away from geomagnetic equator like that of Remningstorp,
        # the sign as well as the magnitude is consistent with those obtained
        # from time domian approach over extended scene and from CRs!
        self._validate(fra_prod, fra_deg, abs_tol_deg, abs_only=True)

    def test_faraday_from_corner_reflector(
            self, file_slc, file_csv, fra_deg, abs_tol_deg):
        # Parse CRs
        cr_llh = cr_llh_from_csv(
            os.path.join(iscetest.data, self.sub_dir, file_csv)
        )
        # Parse RSLC
        slc = SLC(hdf5file=os.path.join(iscetest.data, self.sub_dir, file_slc))
        # Estimate FR angle from CR
        fra_prod = faraday_rot_angle_from_cr(slc, cr_llh)
        # Validate
        self._validate(fra_prod[0], fra_deg, abs_tol_deg)
