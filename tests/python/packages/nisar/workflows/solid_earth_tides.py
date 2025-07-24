import os
import numpy as np

import iscetest
from nisar.workflows import solid_earth_tides

def test_solid_earth_tides_datacube_run():
    '''
    test the solid earth tides datacube
    '''

    # Load the GUNW product from the solid earth tides test data folder
    test_gunw = os.path.join(
        iscetest.data, 'solid_earth_tides','GUNW.h5')

    solid_earth_tides_phase_delay = \
        solid_earth_tides.compute_solid_earth_tides(test_gunw)

    # The solid earth tides only have two layers and should not be equal
    assert solid_earth_tides_phase_delay.ndim == 3
    assert len(solid_earth_tides_phase_delay) == 2
    assert not np.array_equal(solid_earth_tides_phase_delay[0],
                              solid_earth_tides_phase_delay[1])