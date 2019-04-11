// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#include "Phass.h"

/**
 * @param[in] phaseRaster wrapped phase
 * @param[in] corrRaster correlation
 * @param[out] unwRaster unwrapped phase
 * @param[out] labelRaster connected component labels
 */
void isce::unwrap::phass::Phass::
unwrap(isce::io::Raster & phaseRaster,
        isce::io::Raster & corrRaster,
        isce::io::Raster & unwRaster,
        isce::io::Raster & labelRaster)
{
    _usePower = false;
    isce::io::Raster powerRaster("/vsimem/dummy", 1, 1, 1, GDT_Float32, "ENVI");

    std::cout << "unwrapping without intensity" << std::endl;
    unwrap(phaseRaster,
        powerRaster,
       	corrRaster,
        unwRaster,
        labelRaster);
}

/**
* @param[in] phaseRaster wrapped phase
* @param[in] powerRaster power of the interferogram
* @param[in] corrRaster correlation
* @param[out] unwRaster unwrapped phase
* @param[out] labelRaster connected component labels
*/
void isce::unwrap::phass::Phass::
unwrap(isce::io::Raster & phaseRaster,
        isce::io::Raster & powerRaster,
        isce::io::Raster & corrRaster,
        isce::io::Raster & unwRaster,
        isce::io::Raster & labelRaster)
{

    int nrows = phaseRaster.length();
    int ncols = phaseRaster.width();

    float *phase_data_1D = new float[nrows*ncols];
    float *corr_data_1D = new float[nrows*ncols];
    float *power_data_1D = new float[nrows*ncols];

    phaseRaster.getBlock(phase_data_1D, 0, 0, ncols, nrows);
    corrRaster.getBlock(corr_data_1D, 0, 0, ncols, nrows);

    if (_usePower) {
        powerRaster.getBlock(power_data_1D, 0, 0, ncols, nrows);
    }

    float **phase_data  = new float*[nrows];
    float **corr_data   = new float*[nrows]; 
    float **power_data  = new float*[nrows];
        
        
    int **region_map = new int*[nrows];
    for (int line=0; line<nrows; line++){
        region_map[line] = new int[ncols];
    }

    for (int line = 0 ; line < nrows ; ++line) {
        phase_data[line] = &phase_data_1D[line*ncols];
        corr_data[line] = &corr_data_1D[line*ncols];
    }

    if (_usePower) {
        for (int line = 0 ; line < nrows ; ++line) {
            power_data[line] = &power_data_1D[line*ncols];
        }
    } else {
        power_data  = NULL;
    }


    phass_unwrap(nrows, ncols, 
                phase_data, corr_data, power_data, region_map,
                _correlationThreshold, _goodCorrelation,  _minPixelsPerRegion); 

    for (size_t line = 0; line < nrows; ++line) {
        for (size_t col = 0; col < ncols; ++col) {
            phase_data_1D[line*ncols + col] = phase_data[line][col];
        }
    }

    unwRaster.setBlock(phase_data_1D, 0, 0, ncols, nrows);

    float *labels_1D = new float[nrows*ncols];
    for (size_t line = 0; line < nrows; ++line) {
        for (size_t col = 0; col < ncols; ++col) {
            labels_1D[line*ncols + col] = region_map[line][col] + 1;
        }
    }

    labelRaster.setBlock(labels_1D, 0, 0, ncols, nrows);
    
    /*for (int line = 0; line <nrows; ++line) {
        std::cout << line << std::endl;
        delete[] phase_data[line];
        delete[] corr_data[line];
        delete[] power_data[line];
        delete[] region_map[line];
    }*/ 

    delete[] labels_1D;
    delete[] phase_data_1D;
    delete[] power_data_1D;

    delete[] phase_data;
    delete[] corr_data;
    delete[] power_data;
    delete[] region_map;   

}


