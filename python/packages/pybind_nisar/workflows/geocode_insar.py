#!/usr/bin/env python3

"""
collection of functions for NISAR geocode workflow
"""

import time

import h5py
import journal
import pathlib
import pybind_isce3 as isce3
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import h5_prep
from pybind_nisar.workflows.geocode_insar_runconfig import \
    GeocodeInsarRunConfig
from pybind_nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg, runw_hdf5, output_hdf5):
    """
    geocode RUNW products
    """

    # pull parameters from cfg
    ref_hdf5 = cfg["InputFileGroup"]["InputFilePath"]
    freq_pols = cfg["processing"]["input_subset"]["list_of_frequencies"]
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    dem_file = cfg["DynamicAncillaryFileGroup"]["DEMFile"]
    threshold_geo2rdr = cfg["processing"]["geo2rdr"]["threshold"]
    iteration_geo2rdr = cfg["processing"]["geo2rdr"]["maxiter"]
    lines_per_block = cfg["processing"]["blocksize"]["y"]
    dem_block_margin = cfg["processing"]["dem_margin"]
    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    interp_method = cfg["processing"]["geocode"]["interp_method"]
    gunw_datasets = cfg["processing"]["geocode"]["datasets"]
    scratch_path = pathlib.Path(cfg['ProductPathGroup']['ScratchPath'])

    slc = SLC(hdf5file=ref_hdf5)

    info_channel = journal.info("geocode.run")
    info_channel.log("starting geocode")

    # NISAR products are always zero doppler
    grid_zero_doppler = isce3.core.LUT2d()

    # set defaults shared by both frequencies
    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # init geocode object
    geo = isce3.geocode.GeocodeFloat32()

    # init geocode members
    geo.orbit = slc.getOrbit()
    geo.ellipsoid = ellipsoid
    geo.doppler = grid_zero_doppler
    geo.threshold_geo2rdr = threshold_geo2rdr
    geo.numiter_geo2rdr = iteration_geo2rdr
    geo.dem_block_margin = dem_block_margin
    geo.lines_per_block = lines_per_block
    geo.data_interpolator = interp_method

    t_all = time.time()
    with h5py.File(output_hdf5, "a") as dst_h5:
        for freq in freq_pols.keys():
            pol_list = freq_pols[freq]

            radar_grid_slc = slc.getRadarGrid(freq)
            if az_looks > 1 or rg_looks > 1:
                radar_grid_multilook = radar_grid_slc.multilook(az_looks, rg_looks)
            geo_grid = geogrids[freq]
            geo.geogrid(
                geo_grid.start_x,
                geo_grid.start_y,
                geo_grid.spacing_x,
                geo_grid.spacing_y,
                geo_grid.width,
                geo_grid.length,
                geo_grid.epsg,
            )
            src_freq_path = f"/science/LSAR/RUNW/swaths/frequency{freq}/interferogram"
            dst_freq_path = f"/science/LSAR/GUNW/grids/frequency{freq}/interferogram"

            for pol in pol_list:
                src_group_path = f"{src_freq_path}/{pol}"
                dst_group_path = f"{dst_freq_path}/{pol}"

                # iterate over key: dataset name value: bool flag to perform geocode
                for dataset_name, geocode_this_dataset in gunw_datasets.items():
                    if not geocode_this_dataset:
                        continue

                    if (dataset_name == "layoverShadowMask"):
                        # prepare input raster
                        raster_ref = scratch_path / 'rdr2geo' / f'freq{freq}' / 'mask.rdr'
                        input_raster = isce3.io.Raster(str(raster_ref))
                       
                        # access the HDF5 dataset for layover shadow mask
                        geo.data_interpolator = 'NEAREST'
                        radar_grid = radar_grid_slc
                    else:  
                        # prepare input raster
                        input_raster_str = (
                            f"HDF5:{runw_hdf5}:/{src_group_path}/{dataset_name}"
                        )
                        input_raster = isce3.io.Raster(input_raster_str)

                        # access the HDF5 dataset for a given frequency and pol
                        geo.data_interpolator = interp_method
                        radar_grid = radar_grid_multilook

                    dataset_path = f"{dst_group_path}/{dataset_name}"
                    geocoded_dataset = dst_h5[dataset_path]

                    # Construct the output ratster directly from HDF5 dataset
                    geocoded_raster = isce3.io.Raster(
                        f"IH5:::ID={geocoded_dataset.id.id}".encode("utf-8"),
                        update=True,
                    )

                    geo.geocode(
                        radar_grid=radar_grid,
                        input_raster=input_raster,
                        output_raster=geocoded_raster,
                        dem_raster=dem_raster,
                        output_mode=isce3.geocode.GeocodeOutputMode.INTERP
                    )

                    del geocoded_raster

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geocode in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    """
    run geocode from command line
    """

    # load command line args
    geocode_insar_parser = YamlArgparse()
    args = geocode_insar_parser.parse()

    # Get a runconfig dictionary from command line args
    geocode_insar_runconfig = GeocodeInsarRunConfig(args)

    # prepare RIFG HDF5
    out_paths = h5_prep.run(geocode_insar_runconfig.cfg)
    if args.run_config_path is None:
        out_paths['RUNW'] = args.runw_h5
    else:
        out_paths['RUNW'] = geocode_insar_runconfig.cfg['processing']['geocode']['runw_path']

    # Run geocode
    run(geocode_insar_runconfig.cfg, out_paths["RUNW"], out_paths["GUNW"])
