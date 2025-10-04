#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import h5py
import nisar
from nisar.static.doppler import make_native_doppler_lut
from nisar.static.ephemeris import get_cropped_orbit_and_attitude
from nisar.static.geo_grid import get_output_geo_grid
from nisar.static.geometry_layers import compute_geometry_layers
from nisar.static.granule_id import form_granule_id
from nisar.static.layover_shadow_mask import compute_geocoded_layover_shadow_mask
from nisar.static.logging import get_logger, log_elapsed_time
from nisar.static.product import (
    build_hdf5_dataset_creation_kwds_dict,
    populate_grids_group,
    populate_identification_group,
    populate_metadata_group,
)
from nisar.static.rtc_anf_layers import compute_rtc_anf_layers
from nisar.static.runconfig import get_runconfig_params
from nisar.static.util import get_raster_dataset_metadata_item, scratch_directory
from nisar.static.water_mask import binarize_and_reproject_water_mask

import isce3
from isce3.geometry import make_geo_grid_bounding_polygon


def run_static_layers_workflow(config_file: os.PathLike | str) -> None:
    """
    Run the NISAR Static Layers workflow with the specified run configuration file.

    Will generate a single STATIC HDF5 granule, as specified in the runconfig.

    Parameters
    ----------
    config_file : path-like
        The file path to the input STATIC layers runconfig YAML file.
    """
    logger = get_logger()

    logger.info("Begin static layers workflow")

    # Get workflow configuration parameters.
    logger.info("Parse runconfig")
    params = get_runconfig_params(config_file)
    groups = params["runconfig"]["groups"]
    primary_executable_params = groups["primary_executable"]
    dynamic_ancillary_files = groups["dynamic_ancillary_file_group"]
    product_paths = groups["product_path_group"]
    processing_params = groups["processing"]
    output_params = groups["output"]

    # Open the input DEM and water mask raster datasets.
    dem_raster_file = dynamic_ancillary_files["dem_raster_file"]
    water_mask_raster_file = dynamic_ancillary_files["water_mask_raster_file"]
    logger.info(f"Open DEM raster file {dem_raster_file}")
    dem_raster = isce3.io.Raster(dem_raster_file)

    # Construct a DEM interpolator.
    dem_interp_method = processing_params["dem"]["interp_method"]
    dem = isce3.geometry.DEMInterpolator(dem_raster)
    dem.interp_method = dem_interp_method

    # Construct the output geocoded coordinate grid.
    geo_grid_params = processing_params["geo_grid"]
    geo_grid = get_output_geo_grid(dem_raster=dem_raster, **geo_grid_params)
    logger.info(f"Output geo grid: {geo_grid}")

    # Parse the orbit and attitude data from the input XML files. Crop the data to the
    # time interval of interest to avoid possible geo2rdr convergence errors due to
    # ambiguity between orbit periods.
    orbit, attitude = get_cropped_orbit_and_attitude(
        orbit_xml_file=dynamic_ancillary_files["orbit_xml_file"],
        pointing_xml_file=dynamic_ancillary_files["pointing_xml_file"],
        **processing_params["ephemeris"],
    )

    # Get the Doppler centroid associated with the radar grid. NISAR image grids are
    # always zero-Doppler.
    img_grid_doppler = isce3.core.LUT2d()

    # Estimate the required radar grid spacing necessary to avoid undersampling the
    # output geocoded grid.
    # XXX: We deliberately don't pass geo2rdr parameters to either
    # `infer_radar_grid_spacing_from_geo_grid()` or `get_bounding_radar_grid()` because
    # these functions use `geo2rdr_bracket`, which takes different parameters than the
    # legacy `geo2rdr` routine that's used by most of the workflow. Exposing both sets
    # of parameters would introduce a lot of additional bookkeeping for seemingly little
    # benefit.
    logger.info("Estimate maximum required radar grid spacing")
    radar_grid_params = processing_params["radar_grid"]
    look_side = radar_grid_params["look_side"]
    wavelength = radar_grid_params["wavelength"]
    az_spacing, rg_spacing = isce3.geometry.infer_radar_grid_spacing_from_geo_grid(
        geo_grid=geo_grid,
        dem=dem,
        orbit=orbit,
        doppler=img_grid_doppler,
        look_side=look_side,
        wavelength=wavelength,
        **radar_grid_params["spacing"],
    )

    # Compute a radar grid whose footprint on the ground encloses the geocoded grid on
    # which each output layer is defined.
    logger.info("Compute a radar grid spanning the region of interest")
    radar_grid = isce3.geometry.get_bounding_radar_grid(
        geo_grid=geo_grid,
        az_spacing=az_spacing,
        rg_spacing=rg_spacing,
        orbit=orbit,
        look_side=look_side,
        wavelength=wavelength,
        doppler=img_grid_doppler,
        **radar_grid_params["bounding_box"],
    )
    logger.info(f"Using radar grid: {radar_grid}")

    # Get the native Doppler LUT.
    logger.info("Estimate Doppler centroid LUT from ephemeris data")
    native_doppler = make_native_doppler_lut(
        radar_grid=radar_grid,
        orbit=orbit,
        attitude=attitude,
        dem=dem,
        **processing_params["doppler"],
    )

    # Create a (possibly temporary) scratch directory to store intermediate files.
    logger.info("Create scratch directory")
    with scratch_directory(
        product_paths["scratch_dir"], delete=product_paths["delete_scratch_dir"]
    ) as scratch_dir:
        # Compute static geometry layers (height above ellipsoid, line-of-sight X and Y,
        # local incidence angle). Results are stored as GeoTIFF files in the scratch
        # directory.
        logger.info("Compute static geometry layers")
        geo2rdr_params = processing_params["geo2rdr"]
        with log_elapsed_time(logger.info, "Computing static geometry layers"):
            geometry_layers = compute_geometry_layers(
                geo_grid=geo_grid,
                dem_raster=dem_raster,
                orbit=orbit,
                native_doppler=native_doppler,
                look_side=radar_grid.lookside,
                wavelength=radar_grid.wavelength,
                scratch_dir=scratch_dir,
                dem_interp_method=dem_interp_method,
                geo2rdr_params=geo2rdr_params,
            )
            reprojected_dem, los_east, los_north, local_inc_angle = geometry_layers

        # Compute static mask layers (geocoded layover/shadow mask and water mask).
        # Results are stored as GeoTIFF files in the scratch directory.
        logger.info("Compute geocoded layover/shadow mask layer")
        geocode_params = processing_params["geocode"]
        with log_elapsed_time(logger.info, "Computing geocoded layover/shadow mask"):
            layover_shadow_mask = compute_geocoded_layover_shadow_mask(
                radar_grid=radar_grid,
                orbit=orbit,
                img_grid_doppler=img_grid_doppler,
                geo_grid=geo_grid,
                dem_raster=dem_raster,
                scratch_dir=scratch_dir,
                dem_interp_method=dem_interp_method,
                lines_per_block=processing_params["topo"]["lines_per_block"],
                rdr2geo_params=processing_params["rdr2geo"],
                geo2rdr_params=geo2rdr_params,
                memory_mode=geocode_params["memory_mode"],
                min_block_size=geocode_params["min_block_size"],
                max_block_size=geocode_params["max_block_size"],
            )

        logger.info("Compute re-projected binary water mask layer")
        with log_elapsed_time(logger.info, "Computing re-projected binary water mask"):
            binary_water_mask = binarize_and_reproject_water_mask(
                water_distance_raster_file=water_mask_raster_file,
                geo_grid=geo_grid,
                scratch_dir=scratch_dir,
                **processing_params["water_mask"],
            )

        # Compute radiometric terrain correction (RTC) area normalization factor (ANF)
        # layers. Results are stored as GeoTIFF files in the scratch directory.
        logger.info("Compute RTC area normalization factor layers")
        rtc_params = processing_params["rtc"]
        with log_elapsed_time(logger.info, "Computing RTC area normalization layers"):
            gamma0_to_beta0_factor, gamma0_to_sigma0_factor = compute_rtc_anf_layers(
                radar_grid=radar_grid,
                orbit=orbit,
                native_doppler=native_doppler,
                img_grid_doppler=img_grid_doppler,
                geo_grid=geo_grid,
                dem_raster=dem_raster,
                scratch_dir=scratch_dir,
                dem_interp_method=dem_interp_method,
                geo2rdr_params=geo2rdr_params,
                **geocode_params,
                **rtc_params,
            )

        # Infer the orbit pass direction from the orbit velocity vectors.
        orbit_pass_direction = isce3.core.get_orbit_pass_direction(orbit)

        # Pop 'product_counter' from the dict. This parameter is used to form the
        # granule ID but doesn't correspond to any dataset in the 'identification' group
        # of the product. The other dict contents will be passed as keyword arguments to
        # `populate_identification_group()` below.
        product_counter = primary_executable_params.pop("product_counter")

        # Get `validity_start_datetime` from the input parameters as a
        # `datetime.datetime` object. If it was passed as a non-quoted string in ISO
        # 8601 format, `ruamel.yaml` will have already converted it. Otherwise, manually
        # convert it here.
        validity_start_datetime = primary_executable_params.pop(
            "validity_start_datetime"
        )
        if not isinstance(validity_start_datetime, datetime):
            validity_start_datetime = datetime.fromisoformat(validity_start_datetime)

        # Get the unique ID of the granule based on the input parameters.
        radar_band = primary_executable_params["radar_band"]
        geometry_params = groups["geometry"]
        granule_id = form_granule_id(
            mission_id=primary_executable_params["mission_id"],
            radar_band=radar_band,
            product_level=2,
            product_type="STATIC",
            orbit_pass_direction=orbit_pass_direction,
            x_posting=abs(geo_grid.spacing_x),
            y_posting=abs(geo_grid.spacing_y),
            validity_start_datetime=validity_start_datetime,
            composite_release_id=primary_executable_params["composite_release_id"],
            processing_center=primary_executable_params["processing_center"],
            product_counter=product_counter,
            **geometry_params,
        )

        # Get the output HDF5 file name.
        output_hdf5_filename = product_paths["output_hdf5_file"]
        if output_hdf5_filename is None:
            output_hdf5_filename = granule_id + ".h5"

        # Ensure the output directory exists.
        Path(output_hdf5_filename).parent.mkdir(parents=True, exist_ok=True)

        # Create the output HDF5 file.
        logger.info(f"File path of output granule: {output_hdf5_filename}")
        logger.info("Create output HDF5 file")
        with h5py.File(
            output_hdf5_filename,
            mode="w",
            **output_params["file"],
        ) as hdf5_file:
            # Populate global attributes in the root group of the file.
            logger.info("Populate global HDF5 attributes")
            product_spec = nisar.products.get_product_spec("STATIC")
            nisar.products.populate_global_attrs_from_spec(hdf5_file, product_spec)

            # Get the current processing datetime (truncated to integer seconds
            # precision).
            processing_datetime = datetime.now(timezone.utc).replace(microsecond=0)

            # XXX: It's not really obvious what should go in the `zeroDopplerStartTime`
            # and `zeroDopplerEndTime` datasets in the 'identification' group. For now,
            # we'll use the start & stop time of the radar grid, which is roughly
            # analogous what they represent in other NISAR L2 products.
            img_grid_start_datetime = radar_grid.ref_epoch + isce3.core.TimeDelta(
                radar_grid.sensing_start
            )
            img_grid_end_datetime = radar_grid.ref_epoch + isce3.core.TimeDelta(
                radar_grid.sensing_stop
            )

            # Populate the 'identification' group.
            logger.info("Populate identification metadata in output HDF5 file")
            instrument_group = hdf5_file.create_group(f"/science/{radar_band}SAR")
            identification_group = instrument_group.create_group("identification")
            bounding_polygon = make_geo_grid_bounding_polygon(geo_grid, dem=dem)
            populate_identification_group(
                identification_group=identification_group,
                product_spec=product_spec,
                granule_id=granule_id,
                look_direction=radar_grid.lookside,
                orbit_pass_direction=orbit_pass_direction,
                zero_doppler_start_time=img_grid_start_datetime,
                zero_doppler_end_time=img_grid_end_datetime,
                bounding_polygon=bounding_polygon,
                processing_datetime=processing_datetime,
                validity_start_datetime=validity_start_datetime,
                **geometry_params,
                **primary_executable_params,
            )

            # Get DEM and water mask descriptions, if found in the input raster
            # datasets.
            dem_description = get_raster_dataset_metadata_item(
                dem_raster_file, "dem_description", default="(NOT SPECIFIED)"
            )
            water_mask_description = get_raster_dataset_metadata_item(
                water_mask_raster_file,
                "water_mask_description",
                default="(NOT SPECIFIED)",
            )

            # Populate the 'grids' group.
            logger.info("Populate raster layers and grid coordinates in output HDF5")
            grids_group = instrument_group.create_group("STATIC/grids")
            dataset_creation_kwds = build_hdf5_dataset_creation_kwds_dict(
                dataset_shape=(geo_grid.length, geo_grid.width),
                **output_params["dataset"]
            )
            with log_elapsed_time(logger.info, "Writing raster layers to output HDF5"):
                populate_grids_group(
                    grids_group=grids_group,
                    product_spec=product_spec,
                    dataset_creation_kwds=dataset_creation_kwds,
                    reprojected_dem=reprojected_dem,
                    layover_shadow_mask=layover_shadow_mask,
                    local_incidence_angle=local_inc_angle,
                    line_of_sight_x=los_east,
                    line_of_sight_y=los_north,
                    water_mask=binary_water_mask,
                    rtc_gamma_to_sigma_factor=gamma0_to_sigma0_factor,
                    rtc_gamma_to_beta_factor=gamma0_to_beta0_factor,
                    geo_grid=geo_grid,
                    dem_disclaimer=dem_description,
                    water_mask_disclaimer=water_mask_description,
                )

            # Populate the 'metadata' group.
            metadata_group = instrument_group.create_group("STATIC/metadata")
            populate_metadata_group(
                metadata_group=metadata_group,
                product_spec=product_spec,
                orbit=orbit,
                attitude=attitude,
                native_doppler=native_doppler,
                radar_grid=radar_grid,
                software_version=isce3.__version__,
                dem_source=dem_description,
                water_mask_source=water_mask_description,
                runconfig_contents=params,
            )

    logger.info("Done")


def main(args: Sequence[str] | None = None) -> None:
    """
    Main command line entrypoint.

    Parameters
    ----------
    args : sequence of str or None, optional
        The list of arguments. If None, the argument list is taken from `sys.argv`.
        Defaults to None.
    """
    # Setup the argument parser.
    parser = argparse.ArgumentParser(description="Run the NISAR Static Layers workflow")
    parser.add_argument(
        "config_file",
        type=Path,
        help="Run configuration YAML file for the STATIC workflow",
    )

    # Parse the arguments and convert the result to a dict of keyword arguments.
    kwargs = vars(parser.parse_args(args))

    # Run the workflow with the unpacked keyword arguments.
    run_static_layers_workflow(**kwargs)


if __name__ == "__main__":
    main()
