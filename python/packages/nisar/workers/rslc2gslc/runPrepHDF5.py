#!/usr/bin/env python3 #

import h5py
import os
import osr
import numpy as np
from nisar.h5 import cp_h5_meta_data
import isce3

def runPrepHDF5(self):
    '''
    Copies shared data from RSLC HDF5 to GSLC HDF5

    Parameters:
    -----------
    path_src : str
        Full path to source HDF5 file
    path_dst : str
        Full path to destination HDF5 file
    '''

    state = self.state

    # prelim setup
    common_parent_path = 'science/LSAR'
    src_h5 = h5py.File(state.input_hdf5, 'r')

    # rm anything and start from scratch
    try:
        os.remove(state.output_hdf5)
    except FileNotFoundError:
        pass

    dst_h5 = h5py.File(state.output_hdf5, 'w')

    # simple copies of identification, metadata/orbit, metadata/attitude groups
    cp_h5_meta_data(src_h5, dst_h5, os.path.join(common_parent_path, 'identification'))
    cp_h5_meta_data(src_h5, dst_h5, 
                    os.path.join(common_parent_path, 'SLC/metadata/orbit'),
                    os.path.join(common_parent_path, 'GSLC/metadata/orbit'))

    cp_h5_meta_data(src_h5, dst_h5, 
                    os.path.join(common_parent_path, 'SLC/metadata/attitude'),
                    os.path.join(common_parent_path, 'GSLC/metadata/attitude'))

    # copy calibration information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/calibrationInformation'),
            os.path.join(common_parent_path, 'GSLC/metadata/calibrationInformation'),
            excludes=['zeroDopplerTime', 'slantRange'])
                
    # copy processing information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/processingInformation'),
            os.path.join(common_parent_path, 'GSLC/metadata/processingInformation'),
            excludes=['l0bGranules', 'demFiles', 'zeroDopplerTime', 'slantRange'])

    # copy radar grid information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/geolocationGrid'),
            os.path.join(common_parent_path, 'GSLC/metadata/radarGrid'),
            renames={'coordinateX':'xCoordinates',
                'coordinateY':'yCoordinates',
                'zeroDopplerTime':'zeroDopplerAzimuthTime'})

    # copy radar imagery group; assumming shared data
    # XXX option0: to be replaced with actual gslc code
    # XXX option1: do not write GSLC data here; GSLC rasters can be appended to the GSLC HDF5
    for freq in ['A', 'B']:
        ds_ref = os.path.join(common_parent_path, f'SLC/swaths/frequency{freq}')
        if ds_ref not in src_h5:
            continue
        cp_h5_meta_data(src_h5, dst_h5,
                ds_ref,
                os.path.join(common_parent_path, f'GSLC/grids/frequency{freq}'),
                excludes=['acquiredCenterFrequency', 'acquiredAzimuthBandwidth', 
                    'acquiredRangeBandwidth', 'nominalAcquisitionPRF', 'slantRange',
                    'sceneCenterAlongTrackSpacing', 'sceneCenterGroundRangeSpacing',
                    'HH', 'HV', 'VH', 'VV', 'RH', 'RV',
                    'validSamplesSubSwath1', 'validSamplesSubSwath2',
                    'validSamplesSubSwath3', 'validSamplesSubSwath4'],
                renames={'processedCenterFrequency':'centerFrequency',
                    'processedAzimuthBandwidth':'azimuthBandwidth',
                    'processedRangeBandwidth':'rangeBandwidth'})

    
    ctype = h5py.h5t.py_create(np.complex64)
    ctype.commit(dst_h5['/'].id, np.string_('complex64'))

    # create the datasets in the output hdf5
    self.geogrid_dict = {}
    for freq in state.subset_dict.keys():
        frequency = f'frequency{freq}'
        pol_list = state.subset_dict[freq]
        self.geogrid_dict[frequency] = _createGeoGrid(self.userconfig, frequency, src_h5)
        dataset_path = os.path.join(common_parent_path, f'GSLC/grids/{frequency}')
        shape=(self.geogrid_dict[frequency].length, self.geogrid_dict[frequency].width)
        for polarization in pol_list:
            _createDatasets(dst_h5, common_parent_path, 
                    frequency, polarization, shape, chunks=(128, 128))
   
    # adding geogrid and projection information
    for freq in state.subset_dict.keys():
        frequency = f'frequency{freq}'
        _addGeoInformation(dst_h5, common_parent_path, 
                frequency, self.geogrid_dict[frequency])

    dst_h5.close()
    src_h5.close()

def _createGeoGrid(userconfig, frequency, src_h5):
    
    # For production we only fix epsgcode and snap value and will 
    # rely on the rslc product metadta to compute the bounding box of the geocoded products
    # there is a place holder in SLC product for compute Bounding box
    # when that method is populated we should be able to simply say
    # bbox = self.slc_obj.computeBoundingBox(epsg=state.epsg)

    #for now let's rely on the run config input  
    x_start = userconfig['processing']['geocode']['top_left']['x_abs']
    y_start = userconfig['processing']['geocode']['top_left']['y_abs']
    x_end = userconfig['processing']['geocode']['bottom_right']['x_abs']
    y_end = userconfig['processing']['geocode']['bottom_right']['y_abs']
    x_step = userconfig['processing']['geocode']['output_posting'][frequency]['x_posting']
    y_step = -1.0*userconfig['processing']['geocode']['output_posting'][frequency]['y_posting']

    if not x_step:
        print("determine x_step based on input data range bandwidth")
        x_step = _x_step(src_h5, frequency)

    if not y_step:
        y_step = _y_step(src_h5, frequency)

    #top_left_x_snap = userconfig['processing']['geocode']['top_left']['x_snap']
    #top_left_y_snap = userconfig['processing']['geocode']['top_left']['y_snap']
    #bottom_right_x_snap = userconfig['processing']['geocode']['bottom_right']['x_snap']
    #bottom_right_y_snap = userconfig['processing']['geocode']['bottom_right']['y_snap']

    epsg_code = userconfig['processing']['geocode']['output_epsg']


    # snap coordinates
    #x_start = _snap_coordinate(x_start, top_left_x_snap, x_step, np.floor)
    #y_start = _snap_coordinate(y_start, top_left_y_snap, y_step, np.ceil)
    #x_end = _snap_coordinate(x_end, bottom_right_x_snap, x_step, np.ceil)
    #y_end = _snap_coordinate(y_end, bottom_right_y_snap, y_step, np.floor)

    y_size = int(np.round((y_end-y_start)/y_step))
    x_size = int(np.round((x_end-x_start)/x_step))

    geo_grid = isce3.product.geoGridParameters() 
    geo_grid.startX = x_start
    geo_grid.startY = y_start
    geo_grid.spacingX = x_step
    geo_grid.spacingY = y_step
    geo_grid.width = x_size
    geo_grid.length = y_size
    geo_grid.epsg = epsg_code
    
    return geo_grid

def _x_step(src_h5, frequency):
    
    # Posting in east direction for different modes(range bandwidths)
    # based on Table 2-3 in current GSLC spec document
    # x_spacing_dict[rangeBandiwth [MHz]] = east_spacing [meters]
    x_spacing_dict = {5:40, 20:10, 40:5, 80:2.5}

    range_bandwidth = int(src_h5[
        f'science/LSAR/SLC/swaths/{frequency}/processedRangeBandwidth'][()]/1e6)

    x_step = x_spacing_dict.get(range_bandwidth) or x_spacing_dict[
          min(x_spacing_dict.keys(), key = lambda key: abs(key-range_bandwidth))] 
   
    print(f'range bandwidth: {range_bandwidth} MHz')
    print(f'spacing in easting: {x_step} m')

    return x_step

def _y_step(): 

    # Posting in east direction for different modes
    # based on Table 2-3 in current GSLC spec
    y_step = 5.0 # meters
    print(f'spacing in easting: {y_step} m')

    return y_step


def _createDatasets(dst_h5, common_parent_path, frequency, polarization, shape, chunks=(128, 128)):

    print("create empty dataset for frequency: {} polarization: {}".format(frequency, polarization))
    dataset_path = os.path.join(common_parent_path, f'GSLC/grids/{frequency}')
    grp = dst_h5[dataset_path]
    
    ctype = h5py.h5t.py_create(np.complex64)

    if chunks<shape:
        ds = grp.create_dataset(polarization, dtype=ctype, shape=shape, chunks=chunks)
    else:
        ds = grp.create_dataset(polarization, dtype=ctype, shape=shape)

    ds.attrs['description'] = np.string_(
                                      'Geocoded SLC for {} channel'.format(polarization))
    ds.attrs['units'] = np.string_('')
    
    ds.attrs['grid_mapping'] = np.string_("projection")

    return None

# end of file

def _addGeoInformation(hdf5_obj, common_parent_path, frequency, geo_grid):
    
    epsg_code = geo_grid.epsg

    dx = geo_grid.spacingX
    x0 = geo_grid.startX + 0.5*dx
    xf = x0 + geo_grid.width*dx
    x_vect = np.arange(x0, xf, dx, dtype=np.float64)

    dy = geo_grid.spacingY
    y0 = geo_grid.startY + 0.5*dy
    yf = y0 + geo_grid.length*dy
    y_vect = np.arange(y0, yf, dy, dtype=np.float64)

    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
    root_ds = os.path.join(common_parent_path, 'GSLC', 'grids',
                               f'{frequency}')

    # xCoordinates
    h5_ds = os.path.join(root_ds, 'xCoordinates') # float64
    xds = hdf5_obj.create_dataset(h5_ds, data=x_vect)

    # yCoordinates
    h5_ds = os.path.join(root_ds, 'yCoordinates') # float64
    yds = hdf5_obj.create_dataset(h5_ds, data=y_vect)

    xds.make_scale()
    yds.make_scale()  

    #Associate grid mapping with data - projection created later
    h5_ds = os.path.join(root_ds, "projection")

    #Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)

    ###Create a new single int dataset for projections
    projds = hdf5_obj.create_dataset(h5_ds, (), dtype='i')
    projds[()] = epsg_code

    #h5_ds_list.append(h5_ds)

    ##WGS84 ellipsoid
    projds.attrs['semi_major_axis'] = 6378137.0
    projds.attrs['inverse_flattening'] = 298.257223563
    projds.attrs['ellipsoid'] = np.string_("WGS84")

    ##Additional fields
    projds.attrs['epsg_code'] = epsg_code

    ##CF 1.7+ requires this attribute to be named "crs_wkt"
    ##spatial_ref is old GDAL way. Using that for testing only.
    ##For NISAR replace with "crs_wkt"
    projds.attrs['spatial_ref'] = np.string_(srs.ExportToWkt())

    ##Here we have handcoded the attributes for the different cases
    ##Recommended method is to use pyproj.CRS.to_cf() as shown above
    ##To get complete set of attributes.

    ###Geodetic latitude / longitude
    if epsg_code == 4326:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_('latitude_longitude')
            projds.attrs['longitude_of_prime_meridian'] = 0.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("longitude")
            xds.attrs['units'] = np.string_("degrees_east")

            yds.attrs['standard_name'] = np.string_("latitude")
            yds.attrs['units'] = np.string_("degrees_north")

    ### UTM zones
    elif ((epsg_code > 32600 and
               epsg_code < 32661) or
              (epsg_code > 32700 and
               epsg_code < 32761)):
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_('universal_transverse_mercator')
            projds.attrs['utm_zone_number'] = epsg_code % 100 #self.state.output_epsg % 100

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

    ### Polar Stereo North
    elif epsg_code == 3413:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("polar_stereographic")
            projds.attrs['latitude_of_projection_origin'] = 90.0
            projds.attrs['standard_parallel'] = 70.0
            projds.attrs['straight_vertical_longitude_from_pole'] = -45.0
            projds.attrs['false_easting'] = 0.0
            projds.attrs['false_northing'] = 0.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

    ### Polar Stereo south
    elif epsg_code == 3031:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("polar_stereographic")
            projds.attrs['latitude_of_projection_origin'] = -90.0
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0
            projds.attrs['false_easting'] = 0.0
            projds.attrs['false_northing'] = 0.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

    ### EASE 2 for soil moisture L3
    elif epsg_code == 6933:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("lambert_cylindrical_equal_area")
            projds.attrs['longitude_of_central_meridian'] = 0.0
            projds.attrs['standard_parallel'] = 30.0
            projds.attrs['false_easting'] = 0.0
            projds.attrs['false_northing'] = 0.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

    ### Europe Equal Area for Deformation map (to be implemented in isce3)
    elif epsg_code == 3035:
            #Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_("lambert_azimuthal_equal_area")
            projds.attrs['longitude_of_projection_origin']= 10.0
            projds.attrs['latitude_of_projection_origin'] = 52.0
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0
            projds.attrs['false_easting'] = 4321000.0
            projds.attrs['false_northing'] = 3210000.0

            #Setup units for x and y
            xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
            xds.attrs['long_name'] = np.string_("x coordinate of projection")
            xds.attrs['units'] = np.string_("m")

            yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
            yds.attrs['long_name'] = np.string_("y coordinate of projection")
            yds.attrs['units'] = np.string_("m")

    else:
            raise NotImplementedError('Waiting for implementation / Not supported in ISCE3')
