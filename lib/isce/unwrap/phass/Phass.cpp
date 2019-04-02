#include "Phass.h"

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
    powerRaster.getBlock(power_data_1D, 0, 0, ncols, nrows);
    
    float **phase_data  = new float*[nrows];
    float **corr_data   = new float*[nrows];
    float **power_data  = new float*[nrows];

    int **region_map = new int*[nrows];
    for (int line=0; line<nrows; line++){
    //    phase_data[line] = new float[ncols];
    //    corr_data[line] = new float[ncols];
    //    power_data[line] = new float[ncols];
        region_map[line] = new int[ncols];
    }

    /*
    phase_data[0] = phase_data_1D;
    corr_data[0] = corr_data_1D;
    power_data[0] = power_data_1D;

    for (int line=1; line<nrows; line++) {
        phase_data[line] = phase_data_1D[line-1] + ncols;
        corr_data[line] = corr_data_1D[line-1] + ncols;
        power_data[line] = power_data_1D[line-1] + ncols;
    }
    */
    for (int line = 0 ; line < nrows ; ++line) {
        phase_data[line] = &phase_data_1D[line*ncols];
        corr_data[line] = &corr_data_1D[line*ncols];
        power_data[line] = &power_data_1D[line*ncols];
    }

    phass_unwrap(nrows, ncols, 
                phase_data, corr_data, power_data, region_map,
                _corr_th, _good_corr,  _min_pixels_per_region); 

    for (size_t line = 0; line < nrows; ++line) {
        for (size_t col = 0; col < ncols; ++col) {
            phase_data_1D[line*ncols + col] = phase_data[line][col];
        }
    }

    unwRaster.setBlock(phase_data_1D, 0, 0, ncols, nrows);
    //labelRaster.setBlock(&region_map[0], 0, 0, ncols, nrows);
    
    /*
    for (int line = 0; line <nrows; ++line) {
        std::cout << line << std::endl;
        delete[] phase_data[line];
        delete[] corr_data[line];
        delete[] power_data[line];
        delete[] region_map[line];
    }
    */    
    delete[] phase_data; // this needs to be done last
    delete[] corr_data;
    delete[] power_data;
    delete[] region_map;   
}


