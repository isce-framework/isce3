import os
import numpy.testing as npt
import numpy as np

from nisar.cal import est_peak_loc_cr_from_slc, est_cr_az_mid_swath_from_slc
from nisar.products.readers.SLC import SLC
from nisar.workflows.pol_channel_imbalance_from_rslc import cr_llh_from_csv
import iscetest


class TestCornerReflectorSlcFunc:
    # absolute tolerance for angles in rad
    atol_ang = 1e-5
    # absolute tolerance in height in meters
    atol_hgt = 1e-3

    # SLC file with single point target
    file_slc = 'REE_RSLC_out17.h5'

    # CSV file for CR location of "REE_L0B_out17.rdf" in LLH
    file_csv = 'REE_CR_INFO_out17.csv'

    # Target actual location in antenna frame copied from REE input config/RDF
    # file "REE_L0B_out17.rdf", that is EL (deg), AZ (deg).
    cr_el = 0.0
    cr_az = 0.0

    # Parse CSV file for CR location in LLH (rad, rad, m)
    cr_llh = cr_llh_from_csv(os.path.join(iscetest.data, file_csv))

    # Parse slc
    slc = SLC(hdf5file=os.path.join(iscetest.data, file_slc))

    def test_est_peak_loc_cr_from_slc(self):
        list_cr_info = est_peak_loc_cr_from_slc(self.slc, self.cr_llh)

        npt.assert_equal(len(list_cr_info), 1, err_msg='Wrong number of CR!')
        cr = list_cr_info[0]

        # check antenna EL,AZ angles
        npt.assert_allclose(cr.el_ant, self.cr_el, atol=self.atol_ang,
                            err_msg='Wrong EL angle!')
        npt.assert_allclose(cr.az_ant, self.cr_az, atol=self.atol_ang,
                            err_msg='Wrong AZ angle!')
        # check LLH
        npt.assert_allclose(cr.llh[0], self.cr_llh[0, 0], atol=self.atol_ang,
                            err_msg='Wrong longitude!')
        npt.assert_allclose(cr.llh[1], self.cr_llh[0, 1], atol=self.atol_ang,
                            err_msg='Wrong latitude!')
        npt.assert_allclose(cr.llh[2], self.cr_llh[0, 2], atol=self.atol_hgt,
                            err_msg='Wrong height!')

        # check polarization and amplitude
        for pol, amp in cr.amp_pol.items():
            npt.assert_equal(pol, 'HH', err_msg='Wrong polarization!')
            npt.assert_(abs(amp) > 0, msg='Zero amplitude!')

    def test_est_cr_az_mid_swath_from_slc(self):
        az_cr = est_cr_az_mid_swath_from_slc(self.slc)
        cr_llh = cr_llh_from_csv(
            os.path.join(iscetest.data, self.file_csv), az_heading=az_cr
            )
        npt.assert_equal(len(cr_llh), len(self.cr_llh),
                         err_msg=('Some CRs have AZ angle way off from'
                                  f' {np.rad2deg(az_cr)} (deg)!')
                         )
