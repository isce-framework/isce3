from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, overload

import h5py
import numpy as np
from numpy.typing import ArrayLike
from osgeo import ogr, osr

import isce3
import nisar
from isce3.product.cf_conventions import get_grid_mapping_name
from nisar.products import DatasetSpec, ProductSpec
from nisar.products.writers.SLC import quaternion_to_euler  # FIXME: move this function(?)

from .runconfig import dump_runconfig_to_str
from .typing import RunConfigDict
from .util import copy_blockwise


@overload
def to_bytes(s: str) -> np.bytes_:
    ...


@overload
def to_bytes(s: ArrayLike) -> np.ndarray:
    ...


def to_bytes(s):
    """ """
    # XXX TODO FIXME:
    # Traceback (most recent call last):
    #   File "/scratch/gunter/miniforge3/envs/isce3-tmp2/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    #     return _run_code(code, main_globals, None,
    #   File "/scratch/gunter/miniforge3/envs/isce3-tmp2/lib/python3.9/runpy.py", line 87, in _run_code
    #     exec(code, run_globals)
    #   File "/scratch/gunter/projects/tmp2/isce3/build/release/install/packages/nisar/workflows/static.py", line 555, in <module>
    #     main()
    #   File "/scratch/gunter/projects/tmp2/isce3/build/release/install/packages/nisar/workflows/static.py", line 551, in main
    #     run_static_layers_workflow(**parse_args(args))
    #   File "/scratch/gunter/projects/tmp2/isce3/build/release/install/packages/nisar/workflows/static.py", line 517, in run_static_layers_workflow
    #     populate_metadata_group(
    #   File "/scratch/gunter/projects/tmp2/isce3/build/release/install/packages/nisar/static/product.py", line 500, in populate_metadata_group
    #     populate_processing_info_group(
    #   File "/scratch/gunter/projects/tmp2/isce3/build/release/install/packages/nisar/static/product.py", line 460, in populate_processing_info_group
    #     create_processing_info_dataset("runConfigurationContents", runconfig_contents)
    #   File "/scratch/gunter/projects/tmp2/isce3/build/release/install/packages/nisar/static/product.py", line 452, in create_processing_info_dataset
    #     return create_dataset(processing_info_group.file, dataset_spec, data)
    #   File "/scratch/gunter/projects/tmp2/isce3/build/release/install/packages/nisar/static/product.py", line 32, in create_dataset
    #     data = np.asanyarray(data, dtype=dataset_spec.dtype)
    # UnicodeEncodeError: 'ascii' codec can't encode character '\xa9' in position 441: ordinal not in range(128)
    return np.char.encode(s, encoding="utf-8")


def create_dataset(
    hdf5_file: h5py.File,
    dataset_spec: nisar.products.DatasetSpec,
    data: ArrayLike,
    **kwds: Any,
) -> h5py.Dataset:
    """ """
    if np.issubdtype(dataset_spec.dtype, np.bytes_):
        data = to_bytes(data)
    else:
        data = np.asanyarray(data, dtype=dataset_spec.dtype)

    dataset = hdf5_file.create_dataset(name=dataset_spec.name, data=data, **kwds)
    nisar.products.populate_dataset_attrs_from_spec(dataset, dataset_spec)
    return dataset


def look_side_to_str(look_side: isce3.core.LookSide | str) -> str:
    """ """
    look_side = isce3.core.normalize_look_side(look_side)

    if look_side == isce3.core.LookSide.Left:
        return "Left"
    if look_side == isce3.core.LookSide.Right:
        return "Right"

    # Should be unreachable.
    assert False,  f"unexpected look_side {look_side}"


def parse_processing_type_code(code: str) -> str:
    """ """
    if code == "PR":
        return "Nominal"
    if code == "OD":
        return "Custom"

    raise ValueError  # FIXME


def isoformat_integer_seconds(t: datetime) -> str:
    """ """
    if t.microsecond != 0:
        raise ValueError  # FIXME
    return t.isoformat()[:19]


def populate_identification_group(
    identification_group: h5py.Group,
    product_spec: nisar.products.ProductSpec,
    *,
    relative_orbit_number: int,
    frame_number: int,
    mission_id: str,
    platform_name: str,
    instrument_name: str,
    processing_center: str,
    processing_type: str,
    product_type: str,
    granule_id: str,
    product_doi: str,
    product_version: str,
    look_direction: isce3.core.LookSide | str,
    orbit_pass_direction: isce3.core.OrbitPassDirection,
    zero_doppler_start_time: isce3.core.DateTime,
    zero_doppler_end_time: isce3.core.DateTime,
    product_level: str,
    is_geocoded: bool,
    bounding_polygon: ogr.Geometry,
    processing_datetime: datetime,
    validity_start_datetime: datetime,
    composite_release_id: str,
) -> None:
    """ """
    # ...
    def create_identification_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((identification_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        return create_dataset(identification_group.file, dataset_spec, data)

    processing_type = parse_processing_type_code(processing_type)
    look_direction = look_side_to_str(look_direction)
    orbit_pass_direction = str(orbit_pass_direction).capitalize()
    zero_doppler_start_time = zero_doppler_start_time.isoformat()
    zero_doppler_end_time = zero_doppler_end_time.isoformat()
    is_geocoded = str(is_geocoded)
    bounding_polygon = bounding_polygon.ExportToWkt()
    processing_datetime = isoformat_integer_seconds(processing_datetime)
    validity_start_datetime = isoformat_integer_seconds(validity_start_datetime)

    create_identification_dataset("trackNumber", relative_orbit_number)
    create_identification_dataset("frameNumber", frame_number)
    create_identification_dataset("missionId", mission_id)
    create_identification_dataset("platformName", platform_name)
    create_identification_dataset("instrumentName", instrument_name)
    create_identification_dataset("processingCenter", processing_center)
    create_identification_dataset("processingType", processing_type)
    create_identification_dataset("productType", product_type)
    create_identification_dataset("granuleId", granule_id)
    create_identification_dataset("productDoi", product_doi)
    create_identification_dataset("productVersion", product_version)
    create_identification_dataset("productSpecificationVersion", product_spec.version)
    create_identification_dataset("lookDirection", look_direction)
    create_identification_dataset("orbitPassDirection", orbit_pass_direction)
    create_identification_dataset("zeroDopplerStartTime", zero_doppler_start_time)
    create_identification_dataset("zeroDopplerEndTime", zero_doppler_end_time)
    create_identification_dataset("productLevel", product_level)
    create_identification_dataset("isGeocoded", is_geocoded)
    create_identification_dataset("boundingPolygon", bounding_polygon)
    create_identification_dataset("processingDateTime", processing_datetime)
    create_identification_dataset("validityStartDateTime", validity_start_datetime)
    create_identification_dataset("compositeReleaseId", composite_release_id)


def create_uninitialized_dataset(
    hdf5_file: h5py.File,
    dataset_spec: DatasetSpec,
    shape: tuple[int, ...],
    **kwds: Any,
) -> h5py.Dataset:
    """ """
    dataset = hdf5_file.create_dataset(
        name=dataset_spec.name,
        dtype=dataset_spec.dtype,
        shape=shape,
        **kwds,
    )
    nisar.products.populate_dataset_attrs_from_spec(dataset, dataset_spec)
    return dataset


def copy_units_attr_from_dataset_spec(dataset: h5py.Dataset, dataset_spec: DatasetSpec) -> None:
    """ """
    try:
        units = dataset_spec.attrs["units"]
    except KeyError:
        pass
    else:
        dataset.attrs["units"] = units


def get_projection_dataset_attrs_dict(epsg: int) -> dict[str, ArrayLike]:
    """ """
    attrs = {}

    # ...
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    # Attributes for all spatial reference systems.
    attrs["epsg_code"] = epsg
    attrs["spatial_ref"] = to_bytes(srs.ExportToWkt())
    attrs["grid_mapping_name"] = to_bytes(get_grid_mapping_name(srs))
    attrs["semi_major_axis"] = 6378137.0
    attrs["inverse_flattening"] = 298.257223563
    attrs["ellipsoid"] = to_bytes("WGS84")

    if epsg == 4326:
        attrs["longitude_of_prime_meridian"] = 0.0
        return attrs

    # Attributes for all projected spatial reference systems.
    attrs["false_easting"] = srs.GetProjParm(osr.SRS_PP_FALSE_EASTING)
    attrs["false_northing"] = srs.GetProjParm(osr.SRS_PP_FALSE_NORTHING)
    attrs["longitude_of_projection_origin"] = srs.GetProjParm(
        osr.SRS_PP_LONGITUDE_OF_ORIGIN
    )

    # Polar Stereographic (North)
    if epsg == 3413:
        attrs["latitude_of_projection_origin"] = 90.0
        attrs["standard_parallel"] = 70.0
        attrs["straight_vertical_longitude_from_pole"] = -45.0
        return attrs

    # Polar Stereographic (South)
    if epsg == 3031:
        attrs["latitude_of_projection_origin"] = -90.0
        attrs["standard_parallel"] = -71.0
        attrs["straight_vertical_longitude_from_pole"] = 0.0
        return attrs

    # Attribute for non-Polar-Stereographic spatial reference systems.
    attrs["latitude_of_projection_origin"] = srs.GetProjParm(
        osr.SRS_PP_LATITUDE_OF_ORIGIN
    )

    if isce3.core.is_utm(epsg):
        attrs["utm_zone_number"] = epsg % 100
        attrs["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN
        )
        attrs["scale_factor_at_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_SCALE_FACTOR
        )
        return attrs

    # EASE-Grid 2.0
    if epsg == 6933:
        attrs["longitude_of_central_meridian"] = 0.0
        attrs["standard_parallel"] = 30.0
        return attrs

    # LAEA Europe
    if epsg == 3035:
        attrs["standard_parallel"] = -71.0
        attrs["straight_vertical_longitude_from_pole"] = 0.0
        return attrs

    raise NotImplementedError(
        f"EPSG {epsg} waiting for implementation / not supported in ISCE3"
    )


def populate_grids_group(
    grids_group: h5py.Group,
    product_spec: ProductSpec,
    dataset_creation_kwds: Mapping[str, Any],
    *,
    dem: isce3.io.Raster,
    layover_shadow_mask: isce3.io.Raster,
    local_incidence_angle: isce3.io.Raster,
    line_of_sight_x: isce3.io.Raster,
    line_of_sight_y: isce3.io.Raster,
    water_mask: isce3.io.Raster,
    rtc_gamma_to_sigma_factor: isce3.io.Raster,
    rtc_gamma_to_beta_factor: isce3.io.Raster,
    geo_grid: isce3.product.GeoGridParameters,
) -> None:
    """ """
    def create_grids_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((grids_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        dataset = create_dataset(grids_group.file, dataset_spec, data)
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)
        return dataset

    x_spacing = geo_grid.spacing_x
    y_spacing = geo_grid.spacing_y
    x_coords = np.asarray(geo_grid.x_coords) + 0.5 * x_spacing
    y_coords = np.asarray(geo_grid.y_coords) + 0.5 * y_spacing

    create_grids_dataset("xCoordinateSpacing", x_spacing)
    create_grids_dataset("yCoordinateSpacing", y_spacing)
    x_coords_dataset = create_grids_dataset("xCoordinates", x_coords)
    y_coords_dataset = create_grids_dataset("yCoordinates", y_coords)

    proj_dataset = create_grids_dataset("projection", geo_grid.epsg)
    proj_dataset_attrs = get_projection_dataset_attrs_dict(geo_grid.epsg)
    proj_dataset.attrs.update(proj_dataset_attrs)

    # ...
    def create_raster_layer_dataset(name: str, raster: isce3.io.Raster) -> h5py.Dataset:
        # ...
        full_name = "/".join((grids_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)

        # ...
        shape = (raster.length, raster.width)

        # ...
        dataset = create_uninitialized_dataset(
            grids_group.file,
            dataset_spec,
            shape,
            **dataset_creation_kwds,
        )

        # ...
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)

        # Attach dimension scales.
        dataset.dims[0].attach_scale(y_coords_dataset)
        dataset.dims[1].attach_scale(x_coords_dataset)

        # TODO: Check that datatype matches raster datatype exactly.
        # TODO: Add `numpy_dtype` attribute to `isce3.io.Raster`?

        # ...
        copy_blockwise(raster, dataset)

        return dataset

    create_raster_layer_dataset("digitalElevationModel", dem)
    create_raster_layer_dataset("layoverShadowMask", layover_shadow_mask)
    create_raster_layer_dataset("localIncidenceAngle", local_incidence_angle)
    create_raster_layer_dataset("losUnitVectorX", line_of_sight_x)
    create_raster_layer_dataset("losUnitVectorY", line_of_sight_y)
    create_raster_layer_dataset("waterMask", water_mask)
    create_raster_layer_dataset("rtcGammaToSigmaFactor", rtc_gamma_to_sigma_factor)
    create_raster_layer_dataset("rtcGammaToBetaFactor", rtc_gamma_to_beta_factor)


def save_orbit_to_hdf5_group(
    group: h5py.Group,
    product_spec: ProductSpec,
    orbit: isce3.core.Orbit,
) -> None:
    """ """
    orbit.save_to_h5(group)

    for dataset in group.values():
        dataset_spec = product_spec.get_dataset_spec(dataset.name)
        nisar.products.populate_dataset_attrs_from_spec(dataset, dataset_spec)


def save_attitude_to_hdf5_group(
    group: h5py.Group,
    product_spec: ProductSpec,
    attitude: isce3.core.Attitude,
    *,
    orbit: isce3.core.Orbit,
    ellipsoid: isce3.core.Ellipsoid = isce3.core.WGS84_ELLIPSOID,
) -> None:
    """ """
    # ...
    attitude.save_to_h5(group)

    # ...
    for dataset in group.values():
        dataset_spec = product_spec.get_dataset_spec(dataset.name)
        nisar.products.populate_dataset_attrs_from_spec(dataset, dataset_spec)

    # ...
    attitude_type = "Custom"  # FIXME
    dataset_spec = product_spec.get_dataset_spec("/".join([group.name, "attitudeType"]))
    create_dataset(group.file, dataset_spec, attitude_type)

    # XXX TODO FIXME: orbit and attitude must have same reference epoch

    ypr = np.rad2deg([quaternion_to_euler(ti, qi, orbit, ellipsoid)
            for (ti, qi) in zip(attitude.time, attitude.quaternions)])
    dataset_spec = product_spec.get_dataset_spec("/".join([group.name, "eulerAngles"]))
    dataset = create_dataset(group.file, dataset_spec, ypr[:,::-1])
    copy_units_attr_from_dataset_spec(dataset, dataset_spec)


def save_doppler_lut2d_to_hdf5_group(
    group: h5py.Group,
    product_spec: ProductSpec,
    doppler: isce3.core.LUT2d,
    *,
    epoch: isce3.core.DateTime,  # FIXME
) -> None:
    """ """
    def create_doppler_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        dataset = create_dataset(group.file, dataset_spec, data)
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)
        return dataset

    create_doppler_dataset("dopplerCentroid", doppler.data)
    create_doppler_dataset("slantRange", doppler.x_axis)
    create_doppler_dataset("zeroDopplerTime", doppler.y_axis)


def save_radar_grid_to_hdf5_group(
    group: h5py.Group,
    product_spec: ProductSpec,
    radar_grid: isce3.product.RadarGridParameters,
) -> None:
    """ """
    def create_radar_grid_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        dataset = create_dataset(group.file, dataset_spec, data)
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)
        return dataset

    zero_doppler_time_spacing = 1.0 / radar_grid.prf
    center_frequency = isce3.core.speed_of_light / radar_grid.wavelength

    create_radar_grid_dataset("slantRange", radar_grid.slant_ranges)
    create_radar_grid_dataset("slantRangeSpacing", radar_grid.range_pixel_spacing)
    create_radar_grid_dataset("zeroDopplerTime", radar_grid.sensing_times)
    create_radar_grid_dataset("zeroDopplerTimeSpacing", zero_doppler_time_spacing)
    create_radar_grid_dataset("centerFrequency", center_frequency)


def populate_processing_info_group(
    processing_info_group: h5py.Group,
    product_spec: ProductSpec,
    *,
    software_version: str,
    dem_source: str,
    water_mask_source: str,
    config_files: Iterable[os.PathLike | str],
    runconfig_contents: RunConfigDict,
) -> None:
    """ """
    # ...
    def create_processing_info_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((processing_info_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        return create_dataset(processing_info_group.file, dataset_spec, data)

    config_files = list(map(os.fsdecode, config_files))
    runconfig_contents = dump_runconfig_to_str(runconfig_contents)

    create_processing_info_dataset("softwareVersion", software_version)
    create_processing_info_dataset("demSource", dem_source)
    create_processing_info_dataset("waterMaskSource", water_mask_source)
    create_processing_info_dataset("runConfigurationContents", runconfig_contents)


def populate_metadata_group(
    metadata_group: h5py.Group,
    product_spec: ProductSpec,
    *,
    orbit: isce3.core.Orbit,
    attitude: isce3.core.Attitude,
    native_doppler: isce3.core.LUT2d,
    radar_grid: isce3.product.RadarGridParameters,
    software_version: str,
    dem_source: str,
    water_mask_source: str,
    config_files: Iterable[os.PathLike | str],
    runconfig_contents: RunConfigDict,
) -> None:
    """ """
    orbit_group = metadata_group.create_group("orbit")
    save_orbit_to_hdf5_group(orbit_group, product_spec, orbit)

    attitude_group = metadata_group.create_group("attitude")
    save_attitude_to_hdf5_group(attitude_group, product_spec, attitude, orbit=orbit)  # FIXME: pass ellipsoid?

    native_doppler_group = metadata_group.create_group("nativeDoppler")
    save_doppler_lut2d_to_hdf5_group(
        group=native_doppler_group,
        product_spec=product_spec,
        doppler=native_doppler,
        epoch=orbit.reference_epoch,
    )

    radar_grid_group = metadata_group.create_group("radarGridParameters")
    save_radar_grid_to_hdf5_group(
        group=radar_grid_group,
        product_spec=product_spec,
        radar_grid=radar_grid,
    )

    processing_info_group = metadata_group.create_group("processingInformation")
    populate_processing_info_group(
        processing_info_group=processing_info_group,
        product_spec=product_spec,
        software_version=software_version,
        dem_source=dem_source,
        water_mask_source=water_mask_source,
        config_files=config_files,
        runconfig_contents=runconfig_contents,
    )
