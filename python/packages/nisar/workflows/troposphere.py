#!/usr/bin/env python3
import copy
import journal
import os
import pathlib
import time

import h5py
import numpy as np
from osgeo import gdal, osr

import pyaps3 as pa #pyAPS package

from nisar.workflows import h5_prep
from troposphere_runconfig import InsarTroposphereRunConfig
#from nisar.workflows.troposphere_runconfig import InsarTroposphereRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse

def utm_to_wgs84(epsg, x, y):
    '''
    convert the x, y coordinates in UTM projection to WGS84 projection

    epsg: EPSG code of UTM
    x, y are the coordinates in meters
    '''
    #UTM to Lat/Lon
    srs_UTM = osr.SpatialReference()
    srs_UTM.ImportFromEPSG(epsg)

    srs_WGS84 = osr.SpatialReference()
    srs_WGS84.ImportFromEPSG(4326)

    #transformer
    ct_UTM_to_WGS84 = osr.CoordinateTransformation(srs_UTM, srs_WGS84)
    
    #stack the x and y
    x_y_pnts_radar = np.stack(( x.flatten(), y.flatten()), axis=-1)

    #transform to lat/lon
    lat_lon_radar = np.array(ct_UTM_to_WGS84.TransformPoints(x_y_pnts_radar))

    #lat lon of data cube
    lat_datacube = lat_lon_radar[:,0].reshape(x.shape)
    lon_datacube = lat_lon_radar[:,1].reshape(x.shape)

    assert lat_datacube.shape == lon_datacube.shape

    #bounds
    extend = 0.1 #to make sure the weather model cover the entire image

    #extent of the data cube
    snwe = (np.nanmin(lat_datacube)-extend, np.nanmax(lat_datacube) + extend, np.nanmin(lon_datacube) -extend, np.nanmax(lon_datacube) + extend)

    return lat_datacube, lon_datacube, snwe



def run(cfg:dict,gunw_hdf5:str):
    '''
    comput the troposphere delay datacube and added to the  GUNW product

    cfg: troposhere configure dictionary
    gunw_hdf5: GUWN hdf5 file
    '''

    # Create error and info channels
    info_channel = journal.info("troposphere_delay_computation.run")
    info_channel.log("starting insar_troposphere_delay computation")
    
    # Fetch the configurations
    tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']['weather_model']
    tropo_cfg = cfg['processing']['troposphere_delay']
   
    weather_model = cfg['weather_model'].upper()
    reference_weather_model_file = tropo_weather_model_cfg['reference_weather_model_file_path']
    secondary_weather_model_file = tropo_weather_model_cfg['secondary_weather_model_file_path']

    tropo_package = tropo_cfg['package'].lower()
    tropo_delay_direction = tropo_cfg['delay_direction'].lower() 
    tropo_delay_product = tropo_cfg['delay_product'].lower()

    with h5py.File(gunw_hdf5, 'a', libver='latest', swmr=True) as f:
        
        # Fetch the GUWN Incidence Angle Datacube
        ia_cube = np.array(f['science/LSAR/GUNW/metadata/radarGrid/incidenceAngle'])
        xcoord_radar_grid = np.array(f['science/LSAR/GUNW/metadata/radarGrid/xCoordinates'])
        ycoord_radar_grid = np.array(f['science/LSAR/GUNW/metadata/radarGrid/yCoordinates'])
        height_radar_grid = np.array(f['science/LSAR/GUNW/metadata/radarGrid/heightAboveEllipsoid'])

        assert ia_cube.shape == (height_radar_grid.shape[0],ycoord_radar_grid.shape[0],xcoord_radar_grid.shape[0])
        
        # UTM EPSG code
        utm_epsg = int(np.array(f['science/LSAR/GUNW/metadata/radarGrid/epsg']))

        #x and y for the entire datacube
        y_2d_radar = np.tile(ycoord_radar_grid, (len(xcoord_radar_grid), 1)).T
        x_2d_radar = np.tile(xcoord_radar_grid, (len(ycoord_radar_grid), 1))

        # obtain the lat/lon coordinates
        lat_datacube, lon_datacube, _ = utm_to_wgs84(utm_epsg, x_2d_radar, y_2d_radar)

        for delay_product in tropo_delay_product:
            
            # pyaps package
            if tropo_package == 'pyaps':
                
                delay_type = 'dry' if delay_product == 'hydro'
                
                tropo_delay_datacube_list = []
                for index, hgt in enumerate(height_radar_grid):
                    if tropo_delay_direction == 'zenith':
                        inc_datacube = np.zeros(x_2d_radar.shape)
                    else:
                        inc_datacube = ia_cube[index,:,:]
                
                    dem_datacube = np.full(inc_datacube.shape,hgt)
                    
                    #delay for the reference
                    obj_ref = pa.PyAPS(reference_weather_model_file, dem=dem_datacube, inc=inc_datacube, lat=lat_datacube, lon=lon_datacube, grib=weather_model, humidity= 'Q', verb=False, Del=delay_type)
                    phs_ref = obj_ref.getdelay()

                    #delay for the secondary image
                    obj_second = pa.PyAPS(secondary_weather_model_file, dem=dem_datacube, inc=inc_datacube, lat=lat_datacube, lon=lon_datacube, grib=weather_model, humidity= 'Q', verb=False, Del=delay_type)
                    phs_second = obj_second.getdelay()

                    #tropo phase delay
                    tropo_delay_datacube_list.append(phs_ref - phs_second)
                
                #tropo delay datacube
                tropo_delay_datacube = np.stack(tropo_datacube_list)
                tropo_datacube_list = None
                
                assert tropo_delay_datacube.shape == ia_cube.shape
                
                radarGrid = f.get('science/LSAR/GUNW/metadata/radarGrid')
                radarGrid.create_dataset(f'tropoDelay_{tropo_delay_direction}_{delay_product}',tropo_delay_datacube)
            
            # raider package
            else:
                print('raider package is under development currently')


if __name__ == "__main__":
    
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    
    # convert CLI input to run configuration
    tropo_runcfg = InsarTropophereRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(tropo_runcfg.cfg)
    run(tropo_runcfg.cfg, gunw_hdf5=out_paths['GUNW'])
