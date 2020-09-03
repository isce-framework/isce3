#!/usr/bin/env python3
import os
import shutil

import h5py
import numpy as np

from pybind_nisar.workflows import defaults, gslc, h5_prep, runconfig

import iscetest

def test_run():
    '''
    run gslc with same rasters and DEM as geocodeSlc test
    '''
    # load yaml
    test_yaml = os.path.join(iscetest.data, 'geocodeslc/test_gslc.yaml')
    cfg = runconfig.load_yaml(test_yaml, defaults.gslc)

    # set input
    input_h5 = os.path.join(iscetest.data, 'envisat.h5')
    cfg['InputFileGroup']['InputFilePath'] = input_h5

    # reset path to DEM
    dem_path = os.path.join(iscetest.data, 'geocode/zeroHeightDEM.geo')
    cfg['DynamicAncillaryFileGroup']['DEMFile'] = dem_path

    # check and validate semi-valid runconfig. input/output adjusted in loop
    runconfig.prep_paths(cfg)
    runconfig.prep_frequency_and_polarizations(cfg)
    runconfig.prep_geocode_cfg(cfg)

    # geocode same 2 rasters as C++/pybind geocodeSlc
    for xy in ['x', 'y']:
        # adjust runconfig to match just created raster
        cfg['ProductPathGroup']['SASOutputFile'] = f'{xy}_out.h5'

        # prepare output hdf5
        h5_prep.run(cfg, 'SLC', 'GSLC')

        # geocode test raster
        gslc.run(cfg)


if __name__ == '__main__':
    test_run()
