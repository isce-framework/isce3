#!/usr/bin/env python3

import os
import numpy as np
from osgeo import gdal
import isce3.ext.isce3 as isce3
import iscetest
from nisar.products.readers import SLC

# Create list of RadarGridParameters to process
radar_grid_str_list = ['cropped', 'multilooked']

# Create list of rtcAlgorithms
rtc_algorithm_list = [
        isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION,
        isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION]


def test_rtc():

    # Open HDF5 file and create radar grid parameter
    print('iscetest.data:', iscetest.data)
    h5_path = os.path.join(iscetest.data, 'envisat.h5')
    slc_obj = SLC(hdf5file=h5_path)
    frequency = 'A'
    radar_grid_sl = slc_obj.getRadarGrid(frequency)

    # Open DEM raster
    dem_file = os.path.join(iscetest.data, 'srtm_cropped.tif')
    dem_obj = isce3.io.Raster(dem_file)

    # Crop original radar grid parameter
    radar_grid_cropped = \
            radar_grid_sl.offset_and_resize(30, 135, 128, 128)

    # Multi-look original radar grid parameter
    nlooks_az = 5
    nlooks_rg = 5
    radar_grid_ml = \
            radar_grid_sl.multilook(nlooks_az, nlooks_rg)

    # Create orbit and Doppler LUT
    orbit = slc_obj.getOrbit()
    doppler = slc_obj.getDopplerCentroid()
    doppler.bounds_error = False
    # doppler = isce3.core.LUT2d()

    # set input parameters
    input_terrain_radiometry = isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT
    output_terrain_radiometry = isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT

    rtc_area_mode = isce3.geometry.RtcAreaMode.AREA_FACTOR
    rtc_area_beta_mode = isce3.geometry.RtcAreaBetaMode.AUTO

    for radar_grid_str in radar_grid_str_list:

        # Open DEM raster
        if (radar_grid_str == 'cropped'):
            radar_grid = radar_grid_cropped
        else:
            radar_grid = radar_grid_ml

        for rtc_algorithm in rtc_algorithm_list:

            geogrid_upsampling = 1

            # test removed because it requires high geogrid upsampling (too
            # slow)
            if (rtc_algorithm ==
                        isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION and
                radar_grid_str == 'cropped'):
                continue
            elif (rtc_algorithm ==
                       isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION):
                filename = './rtc_bilinear_distribution_' + radar_grid_str + '.bin'
            else:
                filename = './rtc_area_proj_' + radar_grid_str + '.bin'

            print('generating file:', filename)

            # Create output raster
            out_raster = isce3.io.Raster(filename, radar_grid.width,
                                         radar_grid.length, 1, gdal.GDT_Float32,
                                         'ENVI')

            # Call RTC
            isce3.geometry.compute_rtc(radar_grid, orbit, doppler, dem_obj, 
                                       out_raster,
                                       input_terrain_radiometry, 
                                       output_terrain_radiometry,
                                       rtc_area_mode, rtc_algorithm,
                                       rtc_area_beta_mode, geogrid_upsampling)

            del out_raster

    # check results
    for radar_grid_str in radar_grid_str_list:
        for rtc_algorithm in rtc_algorithm_list:

            # test removed because it requires high geogrid upsampling (too
            # slow)
            if (rtc_algorithm ==
                        isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION and
                radar_grid_str == 'cropped'):
                continue
            elif (rtc_algorithm ==
                       isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION):
                max_rmse = 0.7
                filename = './rtc_bilinear_distribution_' + radar_grid_str + '.bin'
            else:
                max_rmse = 0.1
                filename = './rtc_area_proj_' + radar_grid_str + '.bin'

            print('evaluating file:', os.path.abspath(filename))

            # Open computed integrated-area raster
            test_gdal_dataset = gdal.Open(filename)

            # Open reference raster
            ref_filename = os.path.join(
                iscetest.data, 'rtc/rtc_' + radar_grid_str + '.bin')
            
            ref_gdal_dataset = gdal.Open(ref_filename)
            print('reference file:', ref_filename)

            assert(test_gdal_dataset.RasterXSize == ref_gdal_dataset.RasterXSize)
            assert(test_gdal_dataset.RasterYSize == ref_gdal_dataset.RasterYSize)

            square_sum = 0.0 # sum of square difference
            n_nan = 0          # number of NaN pixels
            n_npos = 0          # number of non-positive pixels

            # read test and ref arrays
            test_array = test_gdal_dataset.GetRasterBand(1).ReadAsArray()
            ref_array = ref_gdal_dataset.GetRasterBand(1).ReadAsArray()

            n_valid = 0

            # iterates over rows (i) and columns (j)
            for i in range(ref_gdal_dataset.RasterYSize):
                for j in range(ref_gdal_dataset.RasterXSize):

                    # if nan, increment n_nan
                    if (np.isnan(test_array[i, j]) or np.isnan(ref_array[i, j])): 
                        n_nan = n_nan + 1
                        continue
                    
                    # if n_npos, incremennt n_npos
                    if (ref_array[i, j] <= 0 or test_array[i, j] <= 0):
                        n_npos = n_npos +1
                        continue
                    
                    # otherwise, increment n_valid
                    n_valid = n_valid + 1
                    square_sum += (test_array[i, j] - ref_array[i, j]) ** 2
            print('    ----------------')
            print('    # total:', n_valid + n_nan + n_npos)
            print('    ----------------')
            print('    # valid:', n_valid)
            print('    # NaNs:', n_nan)
            print('    # non-positive:', n_npos)
            print('    ----------------')
            assert(n_valid != 0)

            # Compute average over entire image
            rmse = np.sqrt(square_sum / n_valid)

            print('    RMSE =', rmse)
            print('    ----------------')
            # Enforce bound on average pixel-error
            assert(rmse < max_rmse)

            # Enforce bound on number of ignored pixels
            assert(n_nan < 1e-4 * ref_gdal_dataset.RasterXSize * ref_gdal_dataset.RasterYSize)
            assert(n_npos < 1e-4 * ref_gdal_dataset.RasterXSize * ref_gdal_dataset.RasterYSize)

