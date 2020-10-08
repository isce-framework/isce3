import os

import h5py
import osr

import pybind_isce3 as isce3
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import h5_prep, runconfig


def run(cfg):
    '''
    run geocodeSlc according to parameters in cfg dict
    '''
    # pull parameters from cfg
    input_hdf5 = cfg['InputFileGroup']['InputFilePath']
    output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    geogrids = cfg['processing']['geocode']['geogrids']
    dem_file = cfg['DynamicAncillaryFileGroup']['DEMFile']
    threshold_geo2rdr = cfg['processing']['geo2rdr']['threshold']
    iteration_geo2rdr = cfg['processing']['geo2rdr']['maxiter']
    lines_per_block = cfg['processing']['blocksize']['y']
    dem_block_margin = cfg['processing']['dem_margin']
    flatten = cfg['processing']['flatten']
    ellipsoid = cfg['processing']['geocode']['ellipsoid']

    # init geocodeSLC params
    slc = SLC(hdf5file=input_hdf5)
    orbit = slc.getOrbit()
    dem_raster = isce3.io.Raster(dem_file)

    # Doppler of the image grid (Zero for NISAR)
    image_grid_doppler = isce3.core.LUT2d()

    with h5py.File(output_hdf5, 'a') as dst_h5:
        for freq in freq_pols.keys():
            frequency = f"frequency{freq}"
            pol_list = freq_pols[freq]
            radar_grid = slc.getRadarGrid(freq)
            geo_grid = geogrids[freq]

            for polarization in pol_list:
                # get doppler centroid
                native_doppler = slc.getDopplerCentroid(frequency=freq)

                output_dir = os.path.dirname(os.path.abspath(output_hdf5))
                os.makedirs(output_dir, exist_ok=True)

                raster_ref = f'HDF5:{input_hdf5}:/{slc.slcPath(freq, polarization)}'
                slc_raster = isce3.io.Raster(raster_ref)

                # access the HDF5 dataset for a given frequency and polarization
                dataset_path = f'/science/LSAR/GSLC/grids/{frequency}/{polarization}'
                gslc_dataset = dst_h5[dataset_path]

                # Construct the output ratster directly from HDF5 dataset
                gslc_raster = isce3.io.Raster(f"IH5:::ID={gslc_dataset.id.id}".encode("utf-8"), update=True)

                # run geocodeSlc
                isce3.geocode.geocode_slc(gslc_raster, slc_raster, dem_raster,
                                          radar_grid, geo_grid,
                                          orbit,
                                          native_doppler, image_grid_doppler,
                                          ellipsoid,
                                          threshold_geo2rdr, iteration_geo2rdr,
                                          lines_per_block, dem_block_margin,
                                          flatten)

                # the rasters need to be deleted
                del gslc_raster
                del slc_raster


if __name__ == "__main__":
    cfg = runconfig.load('GSLC')
    h5_prep.run(cfg, 'GSLC')
    run(cfg)
