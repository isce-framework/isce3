import numpy as np
import numpy.testing as npt

import isce3


class TestLLH:
    def test_llh_to_vec3(self):
        lon = np.deg2rad(-118.17130)
        lat = np.deg2rad(34.20150)
        height = 300.0

        llh = isce3.core.LLH(longitude=lon, latitude=lat, height=height)

        npt.assert_array_equal(llh.to_vec3(), [lon, lat, height])
