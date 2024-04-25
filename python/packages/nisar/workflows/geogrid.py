'''
collection of functions for determing and setting geogrid
'''
import copy

import numpy as np
from osgeo import osr
import journal

import isce3
from nisar.products.readers import SLC
from nisar.workflows.stage_dem import margin_km_to_deg

METADATA_CUBES_MARGIN_IN_PIXELS = 5


def _grid_size(stop, start, sz):
    '''
    get grid dim based on start, end, and grid size inputs
    '''
    # check for invalid and return invalid size value
    if None in [stop, start, sz]:
        return np.nan

    return int(np.ceil(np.abs((stop-start)/sz)))


def create(cfg, workflow_name=None, frequency_group=None,
           frequency=None, geocode_dict=None,
           default_spacing_x=None, default_spacing_y=None,
           is_geo_wrapped_igram=False,
           flag_metadata_cubes=False,
           geogrid_ref=None):
    '''
    Create ISCE3 geogrid object based on runconfig parameters

    Parameters
    ----------
    frequency_group: str, optional
        The name of the sub-group that holds the fields x_posting and
        y_posting, which is usually
        the frequency groups "A" or "B". If these fields
        are direct member of the output_posting group, e.g
        for radar_grid_cubes, the frequency_group should be left
        as None.
    frequency: str, optional
        The frequency name, if not provided, it will be
        the same as the frequency_group.
    geocode_dict: dict, optional
        Dictionary to overwrite the default dictionary representing
        the runconfig group processing.geocode
    default_spacing_x: scalar, optional
        Default pixel spacing in the X-direction
    default_spacing_y: scalar, optional
        Default pixel spacing in the Y-direction
    is_geo_wrapped_igram: bool, optional
        Flag indicating if the geogrid is associated with a
        wrapped interferogram
    flag_metadata_cubes: bool, optional
        Flag indicating if the geogrid corresponds to meta data cubes
    geogrid_ref: isce.product.GeoGridParameters, optional
        Geogrid to be used as reference if the runconfig does
        not include geogrid parameter

    Returns
    geogrid: isce.product.GeoGridParameters
        Geogrid object
    '''
    error_channel = journal.error('geogrid.create')

    # unpack and init
    if geocode_dict is None:
        geocode_dict = cfg['processing']['geocode']

    if workflow_name == 'insar':
        input_hdf5 = cfg['input_file_group']['reference_rslc_file']
    else:
        input_hdf5 = cfg['input_file_group']['input_file_path']
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    slc = SLC(hdf5file=input_hdf5)

    # unpack and check cfg dict values. default values set to trigger inside fix(...)
    epsg = geocode_dict['output_epsg']
    start_x = geocode_dict['top_left']['x_abs']
    start_y = geocode_dict['top_left']['y_abs']

    if frequency is None:
        frequency = frequency_group

    # if geocode the wrapped inteferogram
    geo_wrapped_igram = (workflow_name == 'insar') and is_geo_wrapped_igram

    output_posting_group = geocode_dict['output_posting']\
            if not geo_wrapped_igram else geocode_dict['wrapped_interferogram']

    if frequency_group is None:
        spacing_x = output_posting_group['x_posting']
        spacing_y = output_posting_group['y_posting']
    else:
        if geo_wrapped_igram:
            spacing_x = output_posting_group['output_posting']\
                    [frequency_group]['x_posting']
            spacing_y = output_posting_group['output_posting']\
                    [frequency_group]['y_posting']
        else:
            spacing_x = output_posting_group[frequency_group]['x_posting']
            spacing_y = output_posting_group[frequency_group]['y_posting']

    end_x = geocode_dict['bottom_right']['x_abs']
    end_y = geocode_dict['bottom_right']['y_abs']

    assert epsg is not None
    assert 1024 <= epsg <= 32767

    if spacing_y is not None:

        # spacing_y from runconfig should be positive valued
        assert spacing_y > 0.0
        spacing_y = -1.0 * spacing_y

    # copy X spacing from default X spacing (if applicable)
    if spacing_x is None and default_spacing_x is not None:
        spacing_x = default_spacing_x

    # copy Y spacing from default Y spacing (if applicable)
    if spacing_y is None and default_spacing_y is not None:
        spacing_y = default_spacing_y

    epsg_spatial_ref = osr.SpatialReference()
    epsg_spatial_ref.ImportFromEPSG(epsg)

    if spacing_x is None or spacing_y is None:
        dem_raster = isce3.io.Raster(dem_file)

        # Set pixel spacing using the input DEM (same EPSG)
        if epsg == dem_raster.get_epsg():

            # copy X spacing from DEM
            if spacing_x is None:
                spacing_x = dem_raster.dx

                # DEM X spacing should be positive
                if spacing_x <= 0:
                    err_str = f'Expected positive pixel spacing in the X/longitude direction'
                    err_str += f' for DEM {dem_file}. Actual value: {spacing_x}.'
                    error_channel.log(err_str)
                    raise ValueError(err_str)

            # copy Y spacing from DEM
            if spacing_y is None:
                spacing_y = dem_raster.dy

                # DEM Y spacing should be negative
                if spacing_y >= 0:
                    err_str = f'Expected negative pixel spacing in the Y/latitude direction'
                    err_str += f' for DEM {dem_file}. Actual value: {spacing_y}.'
                    error_channel.log(err_str)
                    raise ValueError(err_str)

        else:
            # Set pixel spacing in degrees (lat/lon)
            if epsg_spatial_ref.IsGeographic():
                if spacing_x is None:
                    spacing_x = 0.00017966305682390427
                if spacing_y is None:
                    spacing_y = -0.00017966305682390427

            # Set pixel spacing in meters
            else:
                if spacing_x is None:
                    spacing_x = 20
                if spacing_y is None:
                    spacing_y = -20

    if spacing_x == 0.0 or spacing_y == 0.0:
        err_str = 'spacing_x or spacing_y cannot be 0.0'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # init geogrid
    if None in [start_x, start_y, epsg, end_x, end_y]:

        if geogrid_ref is not None:
            # get grid dimensions for given `spacing_x` and `spacing_y`
            width = _grid_size(geogrid_ref.end_x,
                               geogrid_ref.start_x,
                               spacing_x)
            length = _grid_size(geogrid_ref.end_y,
                                geogrid_ref.start_y,
                                spacing_y)

            # create geogrid based on geogrid_ref with newly computed
            # dimensons
            geogrid = isce3.product.GeoGridParameters(
                start_x=geogrid_ref.start_x,
                start_y=geogrid_ref.start_y,
                spacing_x=spacing_x,
                spacing_y=spacing_y,
                width=width,
                length=length,
                epsg=geogrid_ref.epsg)

            if flag_metadata_cubes:
                geogrid.start_x -= METADATA_CUBES_MARGIN_IN_PIXELS * spacing_x
                geogrid.start_y += (METADATA_CUBES_MARGIN_IN_PIXELS *
                                    abs(spacing_y))
                # Starting coordinates will be snapped to the grid.
                # So, add one extra pixel at the end to make sure that the end
                # coordinates will include the margin defined by
                # METADATA_CUBES_MARGIN_IN_PIXELS
                geogrid.width += 2 * METADATA_CUBES_MARGIN_IN_PIXELS + 1
                geogrid.length += 2 * METADATA_CUBES_MARGIN_IN_PIXELS + 1
        else:
            if flag_metadata_cubes:
                margin = METADATA_CUBES_MARGIN_IN_PIXELS * max([spacing_x,
                                                                spacing_y])
                if epsg_spatial_ref.IsGeographic():
                    margin_in_deg = margin
                else:
                    margin_in_deg = margin_km_to_deg(margin / 1000.0)
            else:
                margin_in_deg = 0

            geogrid = isce3.product.bbox_to_geogrid(
                slc.getRadarGrid(frequency),
                slc.getOrbit(),
                isce3.core.LUT2d(),
                spacing_x, spacing_y, epsg,
                margin=margin_in_deg)

        # restore runconfig start_x (if provided)
        if start_x is not None:
            end_x_from_function = geogrid.start_x + geogrid.spacing_x * geogrid.width
            geogrid.start_x = start_x
            geogrid.width = int(np.ceil((end_x_from_function - start_x) /
                                        geogrid.spacing_x))

        # restore runconfig end_x (if provided)
        if end_x is not None:
            geogrid.width = int(np.ceil((end_x - geogrid.start_x) /
                                        geogrid.spacing_x))

        # restore runconfig start_y (if provided)
        if start_y is not None:
            end_y_from_function = geogrid.start_y + geogrid.spacing_y * geogrid.length
            geogrid.start_y = start_y
            geogrid.length = int(np.ceil((end_y_from_function - start_y) /
                                         geogrid.spacing_y))

        # restore runconfig end_y (if provided)
        if end_y is not None:
            geogrid.length = int(np.ceil((end_y - geogrid.start_y) /
                                         geogrid.spacing_y))

    else:
        width = _grid_size(end_x, start_x, spacing_x)
        length = _grid_size(end_y, start_y, spacing_y)

        # build from probably good user values
        geogrid = isce3.product.GeoGridParameters(start_x, start_y,
                                                  spacing_x, spacing_y,
                                                  width, length, epsg)

    # recheck x+y end points before snap and length+width calculation
    end_pt = lambda start, sz, spacing: start + spacing * sz

    if end_x is None:
        end_x = end_pt(geogrid.start_x, geogrid.spacing_x, geogrid.width)

    if end_y is None:
        end_y = end_pt(geogrid.start_y, geogrid.spacing_y, geogrid.length)

    # snap all the things
    x_snap = geocode_dict['x_snap']
    y_snap = geocode_dict['y_snap']

    # Change the snap if it is to geocode the wrapped interferogram
    if geo_wrapped_igram:
        x_snap = output_posting_group['x_snap']
        y_snap = output_posting_group['y_snap']

    if x_snap is not None or y_snap is not None:
        # check snap values before proceeding
        if x_snap <= 0 or y_snap <= 0:
            err_str = 'Snap values must be > 0.'
            error_channel.log(err_str)
            raise ValueError(err_str)

        if x_snap % spacing_x != 0.0 or y_snap % spacing_y != 0:
            err_str = 'Snap values must be exact multiples of spacings. i.e. snap % spacing == 0.0'
            error_channel.log(err_str)
            raise ValueError(err_str)

        snap_coord = lambda val, snap, round_func: round_func(float(val) / snap) * snap
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)
        end_x = snap_coord(end_x, x_snap, np.ceil)
        end_y = snap_coord(end_y, y_snap, np.floor)
        geogrid.length = _grid_size(end_y, geogrid.start_y, geogrid.spacing_y)
        geogrid.width = _grid_size(end_x, geogrid.start_x, geogrid.spacing_x)

    return geogrid
