#!/usr/bin/env python3
import os
import shutil

import h5py
import numpy as np

import pybind_isce3 as isce3
from pybind_nisar.workflows import defaults, gcov, h5_prep, runconfig

import iscetest

geocode_modes = {'interp':isce3.geocode.GeocodeOutputMode.INTERP,
        'area':isce3.geocode.GeocodeOutputMode.AREA_PROJECTION}
input_axis = ['x', 'y']

def test_run():
    '''
    run gcov with same rasters and DEM as geocodeSlc test
    '''
    # load yaml
    test_yaml = os.path.join(iscetest.data, 'geocode/test_gcov.yaml')
    cfg = runconfig.load_yaml(test_yaml, defaults.gcov)

    # set input
    input_h5 = os.path.join(iscetest.data, 'envisat.h5')
    cfg['InputFileGroup']['InputFilePath'] = input_h5

    # reset path to DEM
    dem_path = os.path.join(iscetest.data, 'geocode/zeroHeightDEM.geo')
    cfg['DynamicAncillaryFileGroup']['DEMFile'] = dem_path
    cfg['ProductPathGroup']['ScratchPath'] = '.'

    # check and validate semi-valid runconfig. input/output adjusted in loop
    runconfig.prep_paths(cfg)
    runconfig.prep_frequency_and_polarizations(cfg)
    runconfig.prep_geocode_cfg(cfg)
    runconfig.prep_gcov(cfg)

    # geocode same rasters as C++/pybind geocodeCov
    for axis in input_axis:
        # adjust runconfig to match just created raster
        cfg['InputFileGroup']['InputFilePath'] = input_h5

        #  iterate thru geocode modes
        for key, value in geocode_modes.items():
            cfg['ProductPathGroup']['SASOutputFile'] = f'{axis}_{key}.h5'

            # prepare output hdf5
            h5_prep.run(cfg, 'SLC', 'GCOV')

            # geocode test raster
            gcov.run(cfg)


if __name__ == '__main__':
    test_run()
