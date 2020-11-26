'''
collection of functions for determing and setting geogrid
'''

import numpy as np

import journal

import pybind_isce3 as isce
from pybind_nisar.products.readers import SLC


def _grid_size(stop, start, sz):
    '''
    get grid dim based on start, end, and grid size inputs
    '''
    # check for invalid and return invalid size value
    if None in [stop, start, sz]:
        return np.nan

    return int(np.ceil(np.abs((stop-start)/sz)))


def create(cfg, frequency):
    '''
    For production we only fix epsgcode and snap value and will
    rely on the rslc product metadta to compute the bounding box of the geocoded products
    there is a place holder in SLC product for compute Bounding box
    when that method is populated we should be able to simply say
    bbox = self.slc_obj.computeBoundingBox(epsg=state.epsg)
    for now let's rely on the run config input
    '''
    error_channel = journal.error('geogrid.create')

    # unpack and init
    geocode_dict = cfg['processing']['geocode']
    input_hdf5 = cfg['InputFileGroup']['InputFilePath']
    dem_file = cfg['DynamicAncillaryFileGroup']['DEMFile']
    slc = SLC(hdf5file=input_hdf5)

    # unpack and check cfg dict values. default values set to trigger inside fix(...)
    epsg = geocode_dict['outputEPSG']
    start_x = geocode_dict['top_left']['x_abs']
    start_y = geocode_dict['top_left']['y_abs']
    spacing_x = geocode_dict['output_posting'][frequency]['x_posting']
    spacing_y = geocode_dict['output_posting'][frequency]['y_posting']
    end_x = geocode_dict['bottom_right']['x_abs']
    end_y = geocode_dict['bottom_right']['y_abs']

    assert epsg is not None
    assert 1024 <= epsg <= 32767

    if spacing_y is not None:

        # spacing_y from runconfig should be positive valued
        assert spacing_y > 0.0
        spacing_y = -1.0 * spacing_y

    # init geogrid
    if None in [start_x, start_y, spacing_x, spacing_y, epsg, end_x, end_y]:

        dem_raster = isce.io.Raster(dem_file)

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

        # extract other geogrid params from radar grid and orbit constructed bounding box
        geogrid = isce.product.bbox_to_geogrid(slc.getRadarGrid(frequency),
                                               slc.getOrbit(),
                                               slc.getDopplerCentroid(frequency=frequency),
                                               spacing_x, spacing_y, epsg)

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
        if spacing_x == 0.0 or spacing_y == 0.0:
            err_str = 'spacing_x or spacing_y cannot be 0.0'
            error_channel.log(err_str)
            raise ValueError(err_str)

        width = _grid_size(end_x, start_x, spacing_x)
        length = _grid_size(end_y, start_y, -1.0*spacing_y)

        # build from probably good user values
        geogrid = isce.product.GeoGridParameters(start_x, start_y,
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
