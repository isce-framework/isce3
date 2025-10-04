from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import h5py
import nisar
import numpy as np
from nisar.products import DatasetSpec, ProductSpec, build_projection_dataset_attrs_dict
from nisar.products.utils import to_bytes
from nisar.products.writers.SLC import quaternion_to_euler  # TODO: move this function
from numpy.typing import ArrayLike
from osgeo import ogr

import isce3

from .runconfig import RunConfigDict, dump_runconfig_to_str
from .util import (
    copy_blockwise,
    isoformat_integer_seconds,
    look_side_to_str,
    parse_processing_type_code,
)


def build_hdf5_dataset_creation_kwds_dict(
    dataset_shape: tuple[int, int],
    *,
    chunk_size: tuple[int, int],
    compression_enabled: bool,
    compression_type: str,
    compression_level: int,
    shuffle: bool,
) -> dict[str, Any]:
    """
    Get keyword parameters to pass to `h5py.Group.create_dataset()`.

    Parameters
    ----------
    dataset_shape : (int, int)
        The shape of the HDF5 dataset. The chunk dimensions will be clipped to avoid
        exceeding the dataset dimensions.
    chunk_size : (int, int)
        Chunk dimensions. Setting `chunk_size` to (-1, -1) will disable chunked storage.
    compression_enabled : bool
        True to enable HDF5 compression.
    compression_type : str
        HDF5 compression filter. Typically one of {'gzip', 'lzf', 'szip'}. See
        https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline for details.
        Ignored if compression is disabled.
    compression_level : int
        Level of compression applied by the GZIP compression filter. Must be an integer
        from 0 to 9 (inclusive). Ignored if compression is disabled or
        `compression_type` is not 'gzip'.
    shuffle : bool
        Enable shuffle filtering during data compression. Ignored if compression is
        disabled.

    Returns
    -------
    dict
        Dict of keyword arguments to pass to `h5py.Group.create_dataset()`.
    """
    # If any chunk dimension exceeds the corresponding dataset dimension, clip it to the
    # dataset dimension.
    chunk_size = tuple(min(a, b) for a, b in zip(chunk_size, dataset_shape))

    kwds = {}
    if chunk_size != (-1, -1):
        kwds["chunks"] = chunk_size
    if compression_enabled:
        kwds["compression"] = compression_type
        kwds["shuffle"] = shuffle
        if compression_type == "gzip":
            kwds["compression_opts"] = compression_level

    return kwds


def create_dataset(
    hdf5_file: h5py.File,
    dataset_spec: DatasetSpec,
    data: ArrayLike,
    **kwds: Any,
) -> h5py.Dataset:
    """
    Create a dataset in an HDF5 file.

    Parameters
    ----------
    hdf5_file : h5py.File
        The HDF5 file.
    dataset_spec : nisar.products.DatasetSpec
        Dataset specifications object describing the name, datatype, and attributes of
        the dataset.
    data : array_like
        The contents of the dataset.
    **kwds
        Additional keyword arguments to pass to `h5py.Group.create_dataset`.

    Returns
    -------
    h5py.Dataset
        The dataset that was created.
    """
    # Convert `data` to a NumPy array with the datatype specified in the spec. If the
    # spec designates the datatype as bytes (i.e. string-valued data), encode the data
    # to a bytestring in 'utf-8' encoding, or an array of such bytestrings.
    if np.issubdtype(dataset_spec.dtype, np.bytes_):
        data = to_bytes(data)
    else:
        data = np.asanyarray(data, dtype=dataset_spec.dtype)

    # Create the dataset.
    dataset = hdf5_file.create_dataset(name=dataset_spec.name, data=data, **kwds)

    # Populate some (not all) attributes from the spec. Note that this only populates
    # common attributes whose values are provided in the spec. Notably, it excludes the
    # 'units' attribute, since its value may depend on a reference epoch. The full set
    # of possible attributes is enumerated in the docstring of
    # `populate_dataset_attrs_from_spec()`.
    nisar.products.populate_dataset_attrs_from_spec(dataset, dataset_spec)

    return dataset


def create_uninitialized_dataset(
    hdf5_file: h5py.File,
    dataset_spec: DatasetSpec,
    shape: tuple[int, ...],
    **kwds: Any,
) -> h5py.Dataset:
    """
    Create a dataset in an HDF5 file without initializing its contents.

    Parameters
    ----------
    hdf5_file : h5py.File
        The HDF5 file.
    dataset_spec : nisar.products.DatasetSpec
        Dataset specifications object describing the name, datatype, and attributes of
        the dataset.
    shape : tuple of int
        The shape of the dataset.
    **kwds
        Additional keyword arguments to pass to `h5py.Group.create_dataset`.

    Returns
    -------
    h5py.Dataset
        The dataset that was created.
    """
    dataset = hdf5_file.create_dataset(
        name=dataset_spec.name,
        dtype=dataset_spec.dtype,
        shape=shape,
        **kwds,
    )
    nisar.products.populate_dataset_attrs_from_spec(dataset, dataset_spec)
    return dataset


def populate_identification_group(
    identification_group: h5py.Group,
    product_spec: ProductSpec,
    *,
    relative_orbit_number: int,
    frame_number: int,
    mission_id: str,
    platform_name: str,
    instrument_name: str,
    radar_band: str,
    processing_center: str,
    processing_type: str,
    granule_id: str,
    product_doi: str,
    product_version: str,
    look_direction: isce3.core.LookSide | str,
    orbit_pass_direction: isce3.core.OrbitPassDirection,
    zero_doppler_start_time: isce3.core.DateTime,
    zero_doppler_end_time: isce3.core.DateTime,
    bounding_polygon: ogr.Geometry,
    processing_datetime: datetime,
    validity_start_datetime: datetime,
    composite_release_id: str,
) -> None:
    """
    Populate the 'identification' group of a NISAR Static Layers product granule.

    Parameters
    ----------
    identification_group : h5py.Group
        The 'identification' group to be populated (e.g.
        '/science/LSAR/identification/').
    product_spec : nisar.products.ProductSpec
        Product specification for the NISAR Static Layers product.
    relative_orbit_number : int
        The relative orbit number (i.e. track number).
    frame_number : int
        The frame number.
    mission_id : str
        The mission ID (e.g. 'NISAR').
    platform_name : str
        The platform name (e.g. 'NISAR').
    instrument_name : str
        The instrument name (e.g. 'L-SAR' or 'S-SAR').
    radar_band : str
        The radar band (e.g. 'L' or 'S').
    processing_center : str
        The processing center (e.g. 'JPL' or 'NRSC').
    processing_type : str
        The processing type code (e.g. 'PR' or 'OD').
    granule_id : str
        The granule ID (excluding any file extension such as '.h5').
    product_doi : str
        The product Digital Object Identifier (DOI).
    product_version : str
        The product version string.
    look_direction : isce3.core.LookSide or str
        The look direction of the sensor (left or right).
    orbit_pass_direction : isce3.core.OrbitPassDirection
        The orbit pass direction (ascending or descending).
    zero_doppler_start_time : isce3.core.DateTime
        Azimuth start time (in UTC) of the granule.
    zero_doppler_end_time : isce3.core.DateTime
        Azimuth stop time (in UTC) of the granule.
    bounding_polygon : osgeo.ogr.Geometry
        OGR polygon bounding the raster data in the granule. Horizontal coordinates
        are longitude followed by latitude (both in degrees), and the vertical
        coordinate is height above the WGS 84 ellipsoid, in meters. The first
        point corresponds to the the starting X and Y coordinate of the raster grid in
        the product's native coordinate system, and the perimeter is traversed in
        counterclockwise order on the ellipsoid. The polygon includes the four corners
        of the raster grid, with equal numbers of points distributed evenly in native
        coordinates of the product along each edge.
    processing_datetime : datetime.datetime
        Processing date and time (in UTC) of the granule, with integer seconds
        precision.
    validity_start_datetime : datetime.datetime
        Starting date and time (in UTC) for when the parameters in the granule are valid
        for the NISAR data acquired, with integer seconds precision.
    composite_release_id : str
        Composite release identifier (CRID) of the product granule.
    """

    # Create a dataset in the 'identification' group.
    def create_identification_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((identification_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        return create_dataset(identification_group.file, dataset_spec, data)

    processing_type = parse_processing_type_code(processing_type)
    look_direction = look_side_to_str(look_direction)
    orbit_pass_direction = str(orbit_pass_direction).capitalize()
    zero_doppler_start_time = zero_doppler_start_time.isoformat()
    zero_doppler_end_time = zero_doppler_end_time.isoformat()
    bounding_polygon = bounding_polygon.ExportToWkt()
    processing_datetime = isoformat_integer_seconds(processing_datetime)
    validity_start_datetime = isoformat_integer_seconds(validity_start_datetime)

    create_identification_dataset("trackNumber", relative_orbit_number)
    create_identification_dataset("frameNumber", frame_number)
    create_identification_dataset("missionId", mission_id)
    create_identification_dataset("platformName", platform_name)
    create_identification_dataset("instrumentName", instrument_name)
    create_identification_dataset("radarBand", radar_band)
    create_identification_dataset("processingCenter", processing_center)
    create_identification_dataset("processingType", processing_type)
    create_identification_dataset("productType", "STATIC")
    create_identification_dataset("granuleId", granule_id)
    create_identification_dataset("productDoi", product_doi)
    create_identification_dataset("productVersion", product_version)
    create_identification_dataset("productSpecificationVersion", product_spec.version)
    create_identification_dataset("lookDirection", look_direction)
    create_identification_dataset("orbitPassDirection", orbit_pass_direction)
    create_identification_dataset("zeroDopplerStartTime", zero_doppler_start_time)
    create_identification_dataset("zeroDopplerEndTime", zero_doppler_end_time)
    create_identification_dataset("productLevel", "L2")
    create_identification_dataset("isGeocoded", "True")
    create_identification_dataset("processingDateTime", processing_datetime)
    create_identification_dataset("validityStartDateTime", validity_start_datetime)
    create_identification_dataset("compositeReleaseId", composite_release_id)

    # The bounding polygon is always in geodetic coordinates w.r.t the WGS 84 ellipsoid
    # (EPSG:4326).
    polygon_dataset = create_identification_dataset("boundingPolygon", bounding_polygon)
    polygon_dataset.attrs["ogr_geometry"] = to_bytes("polygon")
    polygon_dataset.attrs["epsg"] = 4326


def copy_units_attr_from_dataset_spec(
    dataset: h5py.Dataset, dataset_spec: DatasetSpec
) -> None:
    """
    Copy the 'units' attribute from the dataset spec to the dataset, if it exists.

    This has no effect if `dataset_spec` does not contain a 'units' attribute.

    Parameters
    ----------
    dataset : h5py.Dataset
        The HDF5 dataset to copy the 'units' attribute to.
    dataset_spec : nisar.products.DatasetSpec
        The dataset specifications object corresponding to `dataset`.
    """
    try:
        units = dataset_spec.attrs["units"]
    except KeyError:
        pass
    else:
        dataset.attrs["units"] = units


def populate_grids_group(
    grids_group: h5py.Group,
    product_spec: ProductSpec,
    dataset_creation_kwds: Mapping[str, Any],
    *,
    reprojected_dem: isce3.io.Raster,
    layover_shadow_mask: isce3.io.Raster,
    local_incidence_angle: isce3.io.Raster,
    line_of_sight_x: isce3.io.Raster,
    line_of_sight_y: isce3.io.Raster,
    water_mask: isce3.io.Raster,
    rtc_gamma_to_sigma_factor: isce3.io.Raster,
    rtc_gamma_to_beta_factor: isce3.io.Raster,
    geo_grid: isce3.product.GeoGridParameters,
    dem_disclaimer: str,
    water_mask_disclaimer: str,
) -> None:
    """
    Populate the 'grids' group of a NISAR Static Layers product granule.

    Parameters
    ----------
    grids_group : h5py.Group
        The 'grids' group to be populated (e.g. '/science/LSAR/STATIC/grids/').
    product_spec : nisar.products.ProductSpec
        Product specification for the NISAR Static Layers product.
    dataset_creation_kwds : dict
        Dict of keyword arguments to pass to `h5py.Group.create_dataset` when creating
        raster datasets.
    reprojected_dem : isce3.io.Raster
        The re-projected digital elevation model (DEM) raster to store in the granule.
    layover_shadow_mask : isce3.io.Raster
        The layover/shadow mask raster to store in the granule.
    local_incidence_angle : isce3.io.Raster
        The local incidence angle raster to store in the granule.
    line_of_sight_x, line_of_sight_y : isce3.io.Raster
        The X and Y line-of-sight rasters to store in the granule.
    water_mask : isce3.io.Raster
        The water mask raster to store in the granule.
    rtc_gamma_to_sigma_factor : isce3.io.Raster
        The Radiometric Terrain Correction (RTC) gamma0-to-sigma0 area normalization
        factor (ANF) raster to store in the granule.
    rtc_gamma_to_beta_factor : isce3.io.Raster
        The Radiometric Terrain Correction (RTC) gamma0-to-beta0 area normalization
        factor (ANF) raster to store in the granule.
    geo_grid : isce3.product.GeoGridParameters
        The geocoded coordinate grid on which all of the input raster layers are
        sampled.
    dem_disclaimer : str
        A disclaimer string to accompany the DEM dataset in the granule.
    water_mask_disclaimer : str
        A disclaimer string to accompany the water mask dataset in the granule.
    """

    # Create a dataset in the 'grids' group. (This shouldn't be used for creating large
    # raster datasets -- instead use `create_raster_layer_dataset()`, defined below.)
    def create_grids_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((grids_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        dataset = create_dataset(grids_group.file, dataset_spec, data)
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)
        return dataset

    # Get vectors of X and Y coordinates of the center of each pixel in `geo_grid`.
    # (Note that the point (`geo_grid.start_x`, `geo_grid.start_y`) is the upper-left
    # corner of the first pixel -- not the center.)
    x_start = geo_grid.start_x
    y_start = geo_grid.start_y
    x_spacing = geo_grid.spacing_x
    y_spacing = geo_grid.spacing_y
    x_coords = x_start + (0.5 * x_spacing) + x_spacing * np.arange(geo_grid.width)
    y_coords = y_start + (0.5 * y_spacing) + y_spacing * np.arange(geo_grid.length)

    create_grids_dataset("xCoordinateSpacing", x_spacing)
    create_grids_dataset("yCoordinateSpacing", y_spacing)
    x_coords_dataset = create_grids_dataset("xCoordinates", x_coords)
    y_coords_dataset = create_grids_dataset("yCoordinates", y_coords)

    # Treat 'xCoordinates' and 'yCoordinates' as dimension scales. Other NISAR workflows
    # don't assign names to dimension scales so let's take the same approach here.
    x_coords_dataset.make_scale()
    y_coords_dataset.make_scale()

    proj_dataset = create_grids_dataset("projection", geo_grid.epsg)
    proj_dataset_attrs = build_projection_dataset_attrs_dict(geo_grid.epsg)
    proj_dataset.attrs.update(proj_dataset_attrs)

    # Create a dataset in the 'grids' group containing a (potentially large) raster
    # layer. The raster data is copied block-wise into the HDF5 dataset to avoid running
    # out of memory.
    def create_raster_layer_dataset(name: str, raster: isce3.io.Raster) -> h5py.Dataset:
        # Get the full path to the dataset within the HDF5 file. Query the product spec
        # for info about the datatype and attributes of the dataset.
        full_name = "/".join((grids_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)

        # Strict check that the input raster datatype matches the datatype from the
        # product specification exactly. We don't allow implicit narrowing or widening
        # conversions here -- the rasters must be provided in exactly the same datatype
        # that they are stored in.
        if raster.dtype != dataset_spec.dtype:
            raise TypeError(f"{raster.dtype=} must exactly match {dataset_spec.dtype=}")

        # Create the dataset in the HDF5 file without initializing its contents.
        shape = (raster.length, raster.width)
        dataset = create_uninitialized_dataset(
            grids_group.file,
            dataset_spec,
            shape,
            **dataset_creation_kwds,
        )

        # Copy the 'units' attribute from the dataset spec.
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)

        # Attach dimension scales.
        dataset.dims[0].attach_scale(y_coords_dataset)
        dataset.dims[1].attach_scale(x_coords_dataset)

        # Copy contents block-wise from the input raster to the HDF5 dataset.
        copy_blockwise(raster, dataset)

        return dataset

    dem_dataset = create_raster_layer_dataset("digitalElevationModel", reprojected_dem)
    dem_dataset.attrs["disclaimer"] = to_bytes(dem_disclaimer)

    water_mask_dataset = create_raster_layer_dataset("waterMask", water_mask)
    water_mask_dataset.attrs["disclaimer"] = to_bytes(water_mask_disclaimer)

    create_raster_layer_dataset("layoverShadowMask", layover_shadow_mask)
    create_raster_layer_dataset("localIncidenceAngle", local_incidence_angle)
    create_raster_layer_dataset("losUnitVectorX", line_of_sight_x)
    create_raster_layer_dataset("losUnitVectorY", line_of_sight_y)
    create_raster_layer_dataset("rtcGammaToSigmaFactor", rtc_gamma_to_sigma_factor)
    create_raster_layer_dataset("rtcGammaToBetaFactor", rtc_gamma_to_beta_factor)


def save_orbit_to_hdf5_group(
    group: h5py.Group,
    product_spec: ProductSpec,
    orbit: isce3.core.Orbit,
) -> None:
    """
    Populate orbit metadata in an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        The group to serialize orbit metadata to (e.g.
        '/science/LSAR/STATIC/metadata/orbit/').
    product_spec : nisar.products.ProductSpec
        Product specification for the output product containing the orbit metadata.
    orbit : isce3.core.Orbit
        The orbit data to serialize.
    """
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
    """
    Populate attitude metadata in an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        The group to serialize attitude metadata to (e.g.
        '/science/LSAR/STATIC/metadata/attitude/').
    product_spec : nisar.products.ProductSpec
        Product specification for the output product containing the attitude metadata.
    attitude : isce3.core.Attitude
        The attitude data to serialize.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center over a time interval that spans
        the azimuth time extent of `attitude`. Must have the same reference epoch as
        `attitude`.
    ellipsoid : isce3.core.Ellipsoid, optional
        Reference ellipsoid used for converting quaternions to Euler angles, with
        dimensions specified in meters. Defaults to the WGS 84 ellipsoid.
    """
    if orbit.reference_epoch != attitude.reference_epoch:
        raise ValueError(
            "orbit and attitude must have the same reference epoch, got"
            f" {orbit.reference_epoch=} and {attitude.reference_epoch=}"
        )

    # Populate the attitude quaternion data in the group and copy attributes from the
    # dataset specifications.
    attitude.save_to_h5(group)
    for dataset in group.values():
        dataset_spec = product_spec.get_dataset_spec(dataset.name)
        nisar.products.populate_dataset_attrs_from_spec(dataset, dataset_spec)

    # FIXME: There's not currently a good mechanism to get the attitude type. The RSLC
    # workflow hardcodes it to 'Custom' so let's do the same here. See
    # https://github-fn.jpl.nasa.gov/isce-3/isce/issues/2170.
    attitude_type = "Custom"
    dataset_spec = product_spec.get_dataset_spec("/".join([group.name, "attitudeType"]))
    create_dataset(group.file, dataset_spec, attitude_type)

    # Convert the quaternions to Euler angles and store them alongside the quaternion
    # data.
    ypr = np.rad2deg(
        [
            quaternion_to_euler(ti, qi, orbit, ellipsoid)
            for (ti, qi) in zip(attitude.time, attitude.quaternions)
        ]
    )
    dataset_spec = product_spec.get_dataset_spec("/".join([group.name, "eulerAngles"]))
    dataset = create_dataset(group.file, dataset_spec, ypr[:, ::-1])
    copy_units_attr_from_dataset_spec(dataset, dataset_spec)


def add_reference_epoch_units_attr(
    dataset: h5py.Dataset, epoch: isce3.core.DateTime
) -> None:
    """
    Add a 'units' attribute to a dataset describing its reference epoch.

    Creates a 'units' attribute whose contents are a byte string of the form
    b'seconds since YYYY-mm-ddTHH:MM:SS'.

    Parameters
    ----------
    dataset : h5py.Dataset
        The HDF5 dataset.
    epoch : isce3.core.DateTime
        The reference epoch of the time tags in the dataset. Must not contain a
        fractional seconds component.
    """
    if epoch.frac != 0.0:
        raise ValueError(
            "reference epoch must not contain a fractional seconds component; got"
            f" {epoch}"
        )

    date_str = f"{epoch.year:04d}-{epoch.month:02d}-{epoch.day:02d}"
    time_str = f"{epoch.hour:02d}:{epoch.minute:02d}:{epoch.second:02d}"

    dataset.attrs["units"] = to_bytes(f"seconds since {date_str}T{time_str}")


def save_doppler_lut2d_to_hdf5_group(
    group: h5py.Group,
    product_spec: ProductSpec,
    doppler: isce3.core.LUT2d,
    *,
    epoch: isce3.core.DateTime,
) -> None:
    """
    Populate Doppler centroid metadata in an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        The group to serialize the Doppler centroid metadata to (e.g.
        '/science/LSAR/STATIC/metadata/nativeDoppler/').
    product_spec : nisar.products.ProductSpec
        Product specification for the output product containing the Doppler centroid
        metadata.
    doppler : isce3.core.LUT2d
        The trajectory of the radar antenna phase center over a time interval that spans
        the azimuth time extent of `attitude`. Must have the same reference epoch as
        `attitude`.
    epoch : isce3.core.DateTime
        The reference epoch that time tags in the `doppler` lookup table are referenced
        to.
    """

    def create_doppler_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        dataset = create_dataset(group.file, dataset_spec, data)
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)
        return dataset

    range_dataset = create_doppler_dataset("slantRange", doppler.x_axis)

    azimuth_dataset = create_doppler_dataset("zeroDopplerTime", doppler.y_axis)
    add_reference_epoch_units_attr(azimuth_dataset, epoch)

    # Treat 'slantRange' and 'zeroDopplerTime' as dimension scales. Other NISAR
    # workflows don't assign names to dimension scales so let's take the same approach
    # here.
    range_dataset.make_scale()
    azimuth_dataset.make_scale()

    doppler_dataset = create_doppler_dataset("dopplerCentroid", doppler.data)

    # Attach dimension scales.
    doppler_dataset.dims[0].attach_scale(azimuth_dataset)
    doppler_dataset.dims[1].attach_scale(range_dataset)


def save_radar_grid_to_hdf5_group(
    group: h5py.Group,
    product_spec: ProductSpec,
    radar_grid: isce3.product.RadarGridParameters,
) -> None:
    """
    Populate radar grid metadata in an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        The group to serialize orbit metadata to (e.g.
        '/science/LSAR/STATIC/metadata/radarGridParameters/').
    product_spec : nisar.products.ProductSpec
        Product specification for the output product containing the radar grid metadata.
    radar_grid : isce3.product.RadarGridParameters.
        The radar grid parameters to serialize.
    """

    def create_radar_grid_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        dataset = create_dataset(group.file, dataset_spec, data)
        copy_units_attr_from_dataset_spec(dataset, dataset_spec)
        return dataset

    zero_doppler_time_spacing = 1.0 / radar_grid.prf
    center_frequency = isce3.core.speed_of_light / radar_grid.wavelength

    create_radar_grid_dataset("zeroDopplerTimeSpacing", zero_doppler_time_spacing)
    create_radar_grid_dataset("slantRangeSpacing", radar_grid.range_pixel_spacing)
    create_radar_grid_dataset("centerFrequency", center_frequency)

    range_dataset = create_radar_grid_dataset("slantRange", radar_grid.slant_ranges)

    azimuth = radar_grid.sensing_times
    azimuth_dataset = create_radar_grid_dataset("zeroDopplerTime", azimuth)
    add_reference_epoch_units_attr(azimuth_dataset, radar_grid.ref_epoch)

    # Treat 'slantRange' and 'zeroDopplerTime' as dimension scales. Other NISAR
    # workflows don't assign names to dimension scales so let's take the same approach
    # here.
    range_dataset.make_scale()
    azimuth_dataset.make_scale()


def populate_processing_info_group(
    processing_info_group: h5py.Group,
    product_spec: ProductSpec,
    *,
    software_version: str,
    dem_source: str,
    water_mask_source: str,
    runconfig_contents: RunConfigDict,
) -> None:
    """
    Populate the 'processingInformation' group of a NISAR Static Layers product granule.

    Parameters
    ----------
    processing_info_group : h5py.Group
        The 'processingInformation' group to be populated (e.g.
        '/science/LSAR/STATIC/metadata/processingInformation/').
    product_spec : nisar.products.ProductSpec
        Product specification for the NISAR Static Layers product.
    software_version : str
        The version string of the software used to generate the product granule.
    dem_source : str
        A string describing the input digital elevation model (DEM) used in the creation
        of the product granule.
    water_mask_source : str
        A string describing the input water mask used in the creation of the product
        granule.
    runconfig_contents : dict
        A nested dict of run configuration parameters with default values populated.
    """

    # Create a dataset in the 'processingInformation' group.
    def create_processing_info_dataset(name: str, data: ArrayLike) -> h5py.Dataset:
        full_name = "/".join((processing_info_group.name, name))
        dataset_spec = product_spec.get_dataset_spec(full_name)
        return create_dataset(processing_info_group.file, dataset_spec, data)

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
    runconfig_contents: RunConfigDict,
) -> None:
    """
    Populate the 'metadata' group of a NISAR Static Layers product granule.

    Parameters
    ----------
    metadata_group : h5py.Group
        The 'metadata' group to be populated (e.g. '/science/LSAR/STATIC/metadata/').
    product_spec : nisar.products.ProductSpec
        Product specification for the NISAR Static Layers product.
    orbit : isce3.core.Orbit
        The orbit metadata to store in the granule.
    attitude : isce3.core.Attitude
        The attitude metadata to store in the granule.
    native_doppler : isce3.core.LUT2d
        The Doppler centroid metadata to store in the granule.
    radar_grid : isce3.product.RadarGridParameters
        The radar grid parameters to store in the granule.
    software_version : str
        The version string of the software used to generate the product granule.
    dem_source : str
        A string describing the input digital elevation model (DEM) used in the creation
        of the product granule.
    water_mask_source : str
        A string describing the input water mask used in the creation of the product
        granule.
    runconfig_contents : dict
        A nested dict of run configuration parameters with default values populated.
    """
    orbit_group = metadata_group.create_group("orbit")
    save_orbit_to_hdf5_group(orbit_group, product_spec, orbit)

    attitude_group = metadata_group.create_group("attitude")
    save_attitude_to_hdf5_group(attitude_group, product_spec, attitude, orbit=orbit)

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
        runconfig_contents=runconfig_contents,
    )
