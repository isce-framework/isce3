import os
import time

import h5py
import journal

import pybind_isce3 as isce3
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import h5_prep
from pybind_nisar.workflows.yaml_argparse import YamlArgparse
from pybind_nisar.workflows.gslc_runconfig import GSLCRunConfig


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

    # init parameters shared by frequency A and B
    slc = SLC(hdf5file=input_hdf5)
    orbit = slc.getOrbit()
    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Doppler of the image grid (Zero for NISAR)
    image_grid_doppler = isce3.core.LUT2d()

    info_channel = journal.info("gslc.run")
    info_channel.log("starting geocode SLC")

    t_all = time.time()
    with h5py.File(output_hdf5, 'a') as dst_h5:
        for freq in freq_pols.keys():
            frequency = f"frequency{freq}"
            pol_list = freq_pols[freq]
            radar_grid = slc.getRadarGrid(freq)
            geo_grid = geogrids[freq]

            # get doppler centroid
            native_doppler = slc.getDopplerCentroid(frequency=freq)

            for polarization in pol_list:
                t_pol = time.time()

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
                t_pol_elapsed = time.time() - t_pol
                info_channel.log(f'polarization {polarization} ran in {t_pol_elapsed:.3f} seconds')

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran geocode SLC in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    gslc_runcfg = GSLCRunConfig(args)
    h5_prep.run(gslc_runcfg.cfg)
    run(gslc_runcfg.cfg)
