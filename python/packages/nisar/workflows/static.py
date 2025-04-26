#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, TypeVar

import h5py
import numpy as np

import isce3
import nisar
from isce3.core import normalize_data_interp_method
from nisar.products.readers.attitude import load_attitude_from_xml
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.static.bounding_polygon import make_geo_grid_bounding_polygon
from nisar.static.geometry_layers import compute_geometry_layers
from nisar.static.granule_id import form_granule_id
from nisar.static.logging import get_logger, log_elapsed_time
from nisar.static.mask_layers import (
    binarize_and_reproject_water_mask,
    compute_geocoded_layover_shadow_mask,
)
from nisar.static.product import (
    populate_grids_group,
    populate_identification_group,
    populate_metadata_group,
)
from nisar.static.rtc_anf_layers import compute_rtc_anf_layers
from nisar.static.runconfig import (
    get_runconfig_params,
    validate_production_runconfig_params,
)
from nisar.static.util import scratch_directory


T = TypeVar("T")


class XYDict(TypedDict):
    """ """

    x: float | None
    y: float | None


class AzRgDict(TypedDict):
    """ """

    azimuth: float
    range: float


def ceil_divide(num: float, den: float) -> int:
    """ """
    return int(np.ceil(num / den))


def check_dem_coverage(dem: ..., geo_grid: isce3.product.GeoGridParameters) -> None:
    """ """
    # XXX TODO FIXME


def get_output_geo_grid(
    top_left: XYDict,
    bottom_right: XYDict,
    posting: XYDict,
    epsg: int | None,
    dem_raster: isce3.io.Raster,
) -> isce3.product.GeoGridParameters:
    """ """
    # Returns `param` if not None; otherwise `default`.
    def get(param: T | None, *, default: T) -> T:
        return param if (param is not None) else default

    # X and Y coordinate of the upper-left corner pixel of the output geo grid.
    x_ul = get(top_left["x"], default=dem_raster.x0)
    y_ul = get(top_left["y"], default=dem_raster.y0)

    # X and Y spacing of the output geo grid.
    # FIXME: Note the sign flip of user-provided Y posting.
    dx = get(posting["x"], default=dem_raster.dx)
    dy = -posting["y"] if (posting["y"] is not None) else dem_raster.dy

    # X-spacing must be positive and Y-spacing must be negative.
    if not (dx > 0.0) or not (dy < 0.0):
        raise ValueError  # FIXME

    # X and Y coordinate of the lower-right corner pixel of the DEM.
    # dem_x_lr = dem_raster.x0 + dem_raster.dx * (dem_raster.width - 1)  # FIXME: is it correct to subtract 1 from the width/length?
    # dem_y_lr = dem_raster.y0 + dem_raster.dy * (dem_raster.length - 1)
    dem_x_lr = dem_raster.x0 + dem_raster.dx * dem_raster.width
    dem_y_lr = dem_raster.y0 + dem_raster.dy * dem_raster.length

    # X and Y coordinate of the lower-right corner pixel of the output geo grid.
    x_lr = get(bottom_right["x"], default=dem_x_lr)
    y_lr = get(bottom_right["y"], default=dem_y_lr)

    # ...
    if not (x_lr >= x_ul) or not (y_lr <= y_ul):
        raise ValueError  # FIXME

    # TODO: Snap coordinates(?)

    # Length and width of the output geo grid.
    # XXX FIXME TODO: I think we should use `ceil_divide`, but InSAR (i.e. GUNW) seems
    # to use floor division.
    # length = ceil_divide(y_lr - y_ul, dy)
    # width = ceil_divide(x_lr - x_ul, dx)

    def floor_divide(num: float, den: float) -> int:
        return int(np.floor(num / den))

    length = floor_divide(y_lr - y_ul, dy)
    width = floor_divide(x_lr - x_ul, dx)

    # EPSG code of the output geo grid.
    # TODO: If an EPSG was provided, should we re-project DEM to this EPSG code before
    # getting other parameters from it?!
    epsg = get(epsg, default=dem_raster.get_epsg())

    return isce3.product.GeoGridParameters(
        start_x=x_ul,
        start_y=y_ul,
        spacing_x=dx,
        spacing_y=dy,
        width=width,
        length=length,
        epsg=epsg,
    )


def get_cropped_orbit_and_attitude(
    orbit_xml_file: str | os.PathLike,
    pointing_xml_file: str | os.PathLike,
    start_time: str | datetime | None,
    end_time: str | datetime | None,
    padding: float,
) -> tuple[isce3.core.Orbit, isce3.core.Attitude]:
    """


    Notes
    -----
    NISAR orbit and attitude files contain 30 hours of state vectors for 24 hours of
    radar observation data, with 3 hours of padding on either side.

    """
    # ...
    orbit_full = load_orbit_from_xml(orbit_xml_file)
    attitude_full = load_attitude_from_xml(pointing_xml_file)

    # ...
    def get_datetime(
        t: str | datetime | None, *, default: isce3.core.DateTime
    ) -> isce3.core.DateTime:
        if t is None:
            return default
        if isinstance(t, datetime):
            t = t.isoformat()
        return isce3.core.DateTime(t)

    # ...
    start_time = get_datetime(start_time, default=orbit_full.start_datetime)
    end_time = get_datetime(end_time, default=orbit_full.end_datetime)
    padding = isce3.core.TimeDelta(padding)

    # Add padding to start & end times.
    start_time -= padding
    end_time += padding

    # log.info(f"Original {name} data file spans time interval "
    #          f"[{ephemeris.start_datetime}, {ephemeris.end_datetime}]")
    # log.info(f"Cropping {name} to {num_pad} points beyond [{start}, {end}]")

    # Crop orbit. Need at least 4 points for Hermite interpolation.
    orbit_cropped = orbit_full.crop(start_time, end_time, npad=3)

    # Crop attitude. Need at least 2 points for slerp.
    attitude_cropped = attitude_full.crop(start_time, end_time, npad=1)

    # Ensure the orbit & attitude have the same reference epoch.
    epoch = orbit_cropped.reference_epoch
    if attitude_cropped.reference_epoch != epoch:
        attitude_cropped.update_reference_epoch(epoch)

    return orbit_cropped, attitude_cropped


def make_doppler_lut(
    radar_grid: isce3.product.RadarGridParameters,
    orbit: isce3.core.Orbit,
    attitude: isce3.core.Attitude,
    dem: isce3.geometry.DEMInterpolator,
    spacing: AzRgDict,
    interp_method: isce3.core.DataInterpMethod | str,
    bounds_error: bool,
) -> isce3.core.LUT2d:
    """
    """
    # Create a 1-D array with uniform spacing `step` that contains the interval
    # [`start`, `stop`].
    def make_linspace(start, stop, step) -> np.ndarray:
        num = ceil_divide(stop - start, step) + 1
        return start + step * np.arange(num)

    # ...
    start_time = radar_grid.sensing_start
    stop_time = radar_grid.sensing_stop
    az_time = make_linspace(start_time, stop_time, spacing["azimuth"])

    # ...
    near_range = radar_grid.starting_range
    far_range = radar_grid.end_range
    slant_range = make_linspace(near_range, far_range, spacing["range"])

    # ...
    return isce3.geometry.make_doppler_lut_from_attitude(
        az_time=az_time,
        slant_range=slant_range,
        orbit=orbit,
        attitude=attitude,
        wavelength=radar_grid.wavelength,
        dem=dem,
        interp_method=normalize_data_interp_method(interp_method),
        bounds_error=bounds_error,
    )


def build_hdf5_dataset_creation_kwds_dict(
    *,
    chunk_size: tuple[int, int],
    compression_enabled: bool,
    compression_type: str,
    compression_level: int,
    shuffle: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Get keyword parameters to pass to `h5py.Group.create_dataset()`.

    """
    chunk_size = tuple(chunk_size)

    kwds = {}
    if chunk_size != (-1, -1):
        kwds["chunks"] = chunk_size
        if compression_enabled:
            kwds["compression"] = compression_type
            kwds["shuffle"] = shuffle
            if compression_type == "gzip":
                kwds["compression_opts"] = compression_level

    return kwds


def run_static_layers_workflow(config_file: os.PathLike | str) -> None:
    """ """
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

    # ...
    dem_raster = isce3.io.Raster(dynamic_ancillary_files["dem_raster_file"])
    water_mask_raster = isce3.io.Raster(dynamic_ancillary_files["water_mask_raster_file"])

    # Construct the output geographic coordinate grid.
    geo_grid_params = processing_params["geo_grid"]
    geo_grid = get_output_geo_grid(dem_raster=dem_raster, **geo_grid_params)

    # XXX TODO FIXME: check that orbit_and_attitude start time and end time were provided
    processing_type = primary_executable_params["processing_type"]
    processing_center = primary_executable_params["processing_center"]
    dem_source = dynamic_ancillary_files["dem_source_description"]
    water_mask_source = dynamic_ancillary_files["water_mask_source_description"]
    geometry_params = groups["geometry"]
    if processing_type == "PR":
        validate_production_runconfig_params(
            geo_grid=geo_grid,
            product_doi=primary_executable_params["product_doi"],
            processing_center=processing_center,
            dem_source=dem_source,
            water_mask_source=water_mask_source,
            **geometry_params,
        )

    # ...
    logger.info("Read and crop orbit and attitude")
    orbit, attitude = get_cropped_orbit_and_attitude(
        orbit_xml_file=dynamic_ancillary_files["orbit_xml_file"],
        pointing_xml_file=dynamic_ancillary_files["pointing_xml_file"],
        **processing_params["ephemeris"],
    )

    # Get the Doppler centroid associated with the radar grid. NISAR image grids are
    # always zero-Doppler.
    img_grid_doppler = isce3.core.LUT2d()

    # Get the param dict for computing the radar grid and pop out the keys related to
    # the grid azimuth/range spacing and margin.
    radar_grid_params = processing_params["radar_grid"]
    radar_grid_spacing = radar_grid_params.pop("spacing")
    radar_grid_margin = radar_grid_params.pop("margin")

    # Compute a radar grid whose footprint on the ground encloses the geographic grid
    # on which each output layer is defined.
    # XXX We deliberately don't pass geo2rdr parameters here because this function uses
    # `geo2rdr_bracket`, which takes different parameters than the legacy `geo2rdr`
    # routine that's used by most of the workflow. Exposing both sets of parameters
    # would introduce a lot of additional bookkeeping for seemingly little benefit.
    logger.info("Compute a radar grid spanning the region of interest")
    radar_grid = isce3.geometry.get_bounding_radar_grid(
        geo_grid=geo_grid,
        az_spacing=radar_grid_spacing["azimuth"],
        rg_spacing=radar_grid_spacing["range"],
        orbit=orbit,
        doppler=img_grid_doppler,
        az_margin=radar_grid_margin["azimuth"],
        rg_margin=radar_grid_margin["range"],
        **radar_grid_params,
    )

    logger.info(f"Using radar grid: {radar_grid}")

    # ...
    # logger.info("...")
    dem_interp_method = processing_params["dem"]["interp_method"]
    dem = isce3.geometry.DEMInterpolator(dem_raster)
    dem.interp_method = dem_interp_method

    # ...
    logger.info("Create Doppler LUT from spacecraft attitude")
    native_doppler = make_doppler_lut(
        radar_grid=radar_grid,
        orbit=orbit,
        attitude=attitude,
        dem=dem,
        **processing_params["doppler"],
    )

    # ...
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
                radar_grid=radar_grid,
                orbit=orbit,
                native_doppler=native_doppler,
                img_grid_doppler=img_grid_doppler,
                dem_raster=dem_raster,
                geo_grid=geo_grid,
                scratch_dir=scratch_dir,
                dem_interp_method=dem_interp_method,
                geo2rdr_params=geo2rdr_params,
            )

        # Compute static mask layers (geocoded layover/shadow mask and water mask).
        # Results are stored as GeoTIFF files in the scratch directory.
        logger.info("Compute geocoded layover/shadow mask layer")
        geocode_params = processing_params["geocode"]
        with log_elapsed_time(logger.info, "Computing geocoded layover/shadow mask"):
            layover_shadow_mask = compute_geocoded_layover_shadow_mask(
                radar_grid=radar_grid,
                orbit=orbit,
                doppler=img_grid_doppler,
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
                water_mask=water_mask_raster,
                geo_grid=geo_grid,
                scratch_dir=scratch_dir,
                **processing_params["water_mask"],
            )

        # Compute radiometric terrain correction (RTC) area normalization factor (ANF)
        # layers. Results are stored as GeoTIFF files in the scratch directory.
        logger.info("Compute RTC area normalization factor layers")
        rtc_params = processing_params["rtc"]
        with log_elapsed_time(logger.info, "Computing RTC area normalization layers"):
            rtc_anf_layers = compute_rtc_anf_layers(
                radar_grid=radar_grid,
                orbit=orbit,
                native_doppler=native_doppler,
                img_grid_doppler=img_grid_doppler,
                dem_raster=dem_raster,
                geo_grid=geo_grid,
                scratch_dir=scratch_dir,
                dem_interp_method=dem_interp_method,
                geo2rdr_params=geo2rdr_params,
                **geocode_params,
                **rtc_params,
            )

        # ...
        orbit_pass_direction = isce3.core.get_orbit_pass_direction(orbit)

        # Pop 'validity_start_datetime', 'radar_band', and 'product_counter' from the
        # dict. These parameters are used to form the granule ID but don't correspond to
        # any Datasets in the `identification' Group of the product. The other dict
        # contents will be passed as keyword arguments to
        # `populate_identification_group()` below.
        validity_start_datetime = primary_executable_params.pop("validity_start_datetime")
        radar_band = primary_executable_params.pop("radar_band")
        product_counter = primary_executable_params.pop("product_counter")

        # `ruamel.yaml` parses non-quoted datetime-like strings as `datetime.datetime`
        # objects. If that happened, convert it to a datetime string in ISO 8601 format.
        if not isinstance(validity_start_datetime, datetime):
            validity_start_datetime = datetime.fromisoformat(validity_start_datetime)

        # ...
        granule_id = form_granule_id(
            mission_id=primary_executable_params["mission_id"],
            band=radar_band,
            product_level=2,
            product_type="STATIC",
            orbit_pass_direction=orbit_pass_direction,
            x_posting=abs(geo_grid.spacing_x),
            y_posting=abs(geo_grid.spacing_y),
            validity_start_datetime=validity_start_datetime,
            composite_release_id=primary_executable_params["composite_release_id"],
            processing_center=processing_center,
            product_counter=product_counter,
            **geometry_params,
        )

        # ...
        output_hdf5_filename = product_paths["output_hdf5_file"]
        if output_hdf5_filename is None:
            output_hdf5_filename = granule_id + ".h5"

        # ...
        Path(output_hdf5_filename).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"File path of output granule: {output_hdf5_filename}")

        logger.info("Create output HDF5 file")
        with h5py.File(
            output_hdf5_filename,
            mode="w",
            **output_params["file"],
        ) as hdf5_file:
            # ...
            logger.info("Populate global HDF5 attributes")
            product_spec = nisar.products.get_product_spec("STATIC")
            nisar.products.populate_global_attrs_from_spec(hdf5_file, product_spec)

            # ...
            processing_datetime = datetime.now(timezone.utc).replace(microsecond=0)

            # ...
            # XXX: What to use for `zero_doppler_start_time` and
            # `zero_doppler_end_time`?
            logger.info("Populate identification metadata in output HDF5 file")
            instrument_name = primary_executable_params["instrument_name"]
            instrument_group = hdf5_file.create_group(f"/science/{instrument_name}")
            identification_group = instrument_group.create_group("identification")
            bounding_polygon = make_geo_grid_bounding_polygon(geo_grid, dem=dem)
            populate_identification_group(
                identification_group=identification_group,
                product_spec=product_spec,
                product_type="STATIC",
                granule_id=granule_id,
                look_direction=radar_grid.lookside,
                orbit_pass_direction=orbit_pass_direction,
                zero_doppler_start_time=radar_grid.start_datetime,
                zero_doppler_end_time=radar_grid.end_datetime,
                product_level="L2",
                is_geocoded=True,
                bounding_polygon=bounding_polygon,
                processing_datetime=processing_datetime,
                validity_start_datetime=validity_start_datetime,
                **geometry_params,
                **primary_executable_params,
            )

            logger.info("Populate raster layers and grid coordinates in output HDF5 file")
            grids_group = instrument_group.create_group("STATIC/grids")
            dataset_creation_kwds = build_hdf5_dataset_creation_kwds_dict(
                **output_params["dataset"]
            )
            with log_elapsed_time(logger.info, "Writing raster layers to output HDF5"):
                populate_grids_group(
                    grids_group=grids_group,
                    product_spec=product_spec,
                    dataset_creation_kwds=dataset_creation_kwds,
                    dem=geometry_layers.height_above_ellipsoid,
                    layover_shadow_mask=layover_shadow_mask,
                    local_incidence_angle=geometry_layers.local_incidence_angle,
                    line_of_sight_x=geometry_layers.line_of_sight_x,
                    line_of_sight_y=geometry_layers.line_of_sight_y,
                    water_mask=binary_water_mask,
                    rtc_gamma_to_sigma_factor=rtc_anf_layers.gamma0_to_sigma0_factor,
                    rtc_gamma_to_beta_factor=rtc_anf_layers.gamma0_to_beta0_factor,
                    geo_grid=geo_grid,
                )

            metadata_group = instrument_group.create_group("STATIC/metadata")
            populate_metadata_group(
                metadata_group=metadata_group,
                product_spec=product_spec,
                orbit=orbit,
                attitude=attitude,
                native_doppler=native_doppler,
                radar_grid=radar_grid,
                software_version=isce3.__version__,
                dem_source=dem_source,
                water_mask_source=water_mask_source,
                config_files=[config_file],
                runconfig_contents=params,
            )

    logger.info("Done")


def setup_arg_parser() -> argparse.ArgumentParser:
    """ """
    parser = argparse.ArgumentParser(description="")  # FIXME
    parser.add_argument("config_file", type=Path, help="")  # FIXME
    return parser


def parse_args(args: Sequence[str] | None = None) -> dict[str, Any]:
    """ """
    parser = setup_arg_parser()
    params = parser.parse_args(args)
    return vars(params)


def main(args: Sequence[str] | None = None) -> None:
    """ """
    run_static_layers_workflow(**parse_args(args))


if __name__ == "__main__":
    main()
