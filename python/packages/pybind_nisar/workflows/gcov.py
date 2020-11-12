'''
collection of functions for NISAR GCOV workflow
'''

import os
import tempfile
import time

import gdal
import h5py
import journal
import numpy as np

import pybind_isce3 as isce
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import h5_prep, runconfig

def run(cfg):
    '''
    run GCOV
    '''

    # pull parameters from cfg
    input_hdf5 = cfg['InputFileGroup']['InputFilePath']
    output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    scratch_path = cfg['ProductPathGroup']['ScratchPath']

    dem_file = cfg['DynamicAncillaryFileGroup']['DEMFile']
    dem_margin = cfg['processing']['dem_margin']

    # unpack geocode run parameters
    geocode_dict = cfg['processing']['geocode']
    output_mode = geocode_dict['algorithm_type']
    memory_mode = geocode_dict['memory_mode']
    geogrid_upsampling = geocode_dict['geogrid_upsampling']
    abs_cal_factor = geocode_dict['abs_rad_cal']
    geogrids = geocode_dict['geogrids']
    ellipsoid = geocode_dict['ellipsoid']

    # unpack RTC run parameters
    rtc_dict = cfg['processing']['rtc']
    rtc_algorithm = rtc_dict['algorithm_type']
    input_terrain_radiometry = rtc_dict['input_terrain_radiometry']
    rtc_min_value_db = rtc_dict['rtc_min_value_db']
    apply_rtc = output_mode == isce.geocode.GeocodeOutputMode.AREA_PROJECTION_GAMMA_NAUGHT

    # unpack geo2rdr parameters
    geo2rdr_dict = cfg['processing']['geo2rdr']
    threshold = geo2rdr_dict['threshold']
    maxiter = geo2rdr_dict['maxiter']

    if apply_rtc:
        output_radiometry_str = "gamma-naught"
    elif input_terrain_radiometry == isce.geometry.RtcInputRadiometry.BETA_NAUGHT:
        output_radiometry_str = 'beta-naught'
    else:
        output_radiometry_str = 'sigma-naught-ellipsoid'

    # unpack pre-processing
    preprocess = cfg['processing']['pre_process']
    radar_grid_nlooks = preprocess['range_looks'] * preprocess['azimuth_looks']

    # init parameters shared between frequencyA and frequencyB sub-bands
    slc = SLC(hdf5file=input_hdf5)
    dem_raster = isce.io.Raster(dem_file)
    zero_doppler = isce.core.LUT2d()

    exponent = 2
    output_dtype = gdal.GDT_Float32

    info_channel = journal.info("gcov.run")
    error_channel = journal.error("gcov.run")
    info_channel.log("starting geocode COV")

    # XXX only do diagonal elements as spec'd?
    t_all = time.time()
    for frequency in freq_pols.keys():
        t_freq = time.time()

        # unpack frequency dependent parameters
        radar_grid = slc.getRadarGrid(frequency)
        geogrid = geogrids[frequency]
        pol_list = freq_pols[frequency]

        # do no processing if no polarizations specified for current frequency
        if not pol_list:
            continue

        # construct input rasters
        input_raster_list = []
        for pol in pol_list:
            raster_ref = f'HDF5:"{input_hdf5}":/{slc.slcPath(frequency, pol)}'
            temp_raster = isce.io.Raster(raster_ref)
            input_raster_list.append(temp_raster)

        # set paths temporary files
        input_temp = tempfile.NamedTemporaryFile(dir=scratch_path, suffix='.vrt')
        input_raster_obj = isce.io.Raster(input_temp.name, raster_list=input_raster_list)

        nbands = input_raster_obj.num_bands

        # init Geocode object according to the raster type
        if input_raster_obj.datatype() == gdal.GDT_Float32:
            geo = isce.geocode.GeocodeFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_Float64:
            geo = isce.geocode.GeocodeFloat64()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat32:
            geo = isce.geocode.GeocodeCFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat64:
            geo = isce.geocode.GeocodeCFloat64()
        else:
            err_str = 'Unsupported raster type for geocoding'
            error_channel.log(err_str)
            raise NotImplementedError(err_str)

        # init geocode members
        geo.orbit = slc.getOrbit()
        geo.ellipsoid = ellipsoid
        geo.doppler = zero_doppler
        geo.threshold_geo2rdr = threshold
        geo.num_iter_geo2rdr = maxiter
        geo.dem_block_margin = dem_margin

        geo.geogrid(geogrid.start_x, geogrid.start_y,
                geogrid.spacing_x, geogrid.spacing_y,
                geogrid.width, geogrid.length, geogrid.epsg)

        if geogrid_upsampling is None:
            geogrid_upsampling = 1

        # create output raster
        temp_output = tempfile.NamedTemporaryFile(dir=scratch_path, suffix='.tiff')
        output_raster_obj = isce.io.Raster(temp_output.name,
                geogrid.width, geogrid.length, input_raster_obj.num_bands,
                gdal.GDT_Float32, 'GTiff')

        # geocode rasters
        geo.geocode(radar_grid=radar_grid,
                    input_raster=input_raster_obj,
                    output_raster=output_raster_obj,
                    dem_raster=dem_raster,
                    output_mode=output_mode,
                    geogrid_upsampling=geogrid_upsampling,
                    input_radiometry=input_terrain_radiometry,
                    exponent=exponent,
                    rtc_min_value_db=rtc_min_value_db,
                    rtc_algorithm=rtc_algorithm,
                    abs_cal_factor=abs_cal_factor,
                    clip_min = 0, # backscatter lower linear bound to screen negative values
                    clip_max = 2, # backscatter upper linear bound, ~3dB
                    radargrid_nlooks=radar_grid_nlooks,
                    input_rtc=None,
                    output_rtc=None,
                    mem_mode=memory_mode)

        del output_raster_obj

        with h5py.File(output_hdf5, 'a') as hdf5_obj:
            hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
            root_ds = f'/science/LSAR/GCOV/grids/frequency{frequency}'

            h5_ds = os.path.join(root_ds, 'listOfPolarizations')
            if h5_ds in hdf5_obj:
                del hdf5_obj[h5_ds]
            pol_list_s2 = np.array(pol_list, dtype='S2')
            dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
            dset.attrs['description'] = np.string_(
                'List of processed polarization layers with frequency ' +
                frequency)

            h5_ds = os.path.join(root_ds, 'radiometricTerrainCorrectionFlag')
            if h5_ds in hdf5_obj:
                del hdf5_obj[h5_ds]
            dset = hdf5_obj.create_dataset(h5_ds, data=bool(apply_rtc))

            # save GCOV diagonal elements
            xds = hdf5_obj[os.path.join(root_ds, 'xCoordinates')]
            yds = hdf5_obj[os.path.join(root_ds, 'yCoordinates')]
            cov_elements_list = [p.upper()+p.upper() for p in pol_list]
            _save_hdf5_dataset(temp_output.name, hdf5_obj, root_ds,
                               yds, xds, cov_elements_list,
                               standard_name=output_radiometry_str,
                               long_name=output_radiometry_str,
                               units='unitless',
                               fill_value=np.nan,
                               valid_min=0,
                               valid_max=2)
        t_freq_elapsed = time.time() - t_freq
        info_channel.log(f'frequency {frequency} ran in {t_freq_elapsed:.3f} seconds')

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran geocode COV in {t_all_elapsed:.3f} seconds")


def _save_hdf5_dataset(ds_filename, h5py_obj, root_path,
                       yds, xds, ds_name, standard_name=None,
                       long_name=None, units=None, fill_value=None,
                       valid_min=None, valid_max=None):
    '''
    write temporary raster file contents to HDF5

    Parameters
    ----------
    ds_filename : string
        source raster file
    h5py_obj : h5py object
        h5py object of destination HDF5
    root_path : string
        path of output raster data
    yds : h5py dataset object
        y-axis dataset
    xds : h5py dataset object
        x-axis dataset
    ds_name : string
        name of dataset to be added to root_path
    standard_name : string, optional
    long_name : string, optional
    units : string, optional
    fill_value : float, optional
    valid_min : float, optional 
    valid_max : float, optional
    '''
    if not os.path.isfile(ds_filename):
        return

    gdal_ds = gdal.Open(ds_filename)
    nbands = gdal_ds.RasterCount
    for band in range(nbands):
        data = gdal_ds.GetRasterBand(band+1).ReadAsArray()

        if isinstance(ds_name, str):
            h5_ds = os.path.join(root_path, ds_name)
        else:
            h5_ds = os.path.join(root_path, ds_name[band-1])

        if h5_ds in h5py_obj:
            del h5py_obj[h5_ds]

        dset = h5py_obj.create_dataset(h5_ds, data=data)
        dset.dims[0].attach_scale(yds)
        dset.dims[1].attach_scale(xds)
        dset.attrs['grid_mapping'] = np.string_("projection")

        if standard_name is not None:
            dset.attrs['standard_name'] = np.string_(standard_name)

        if long_name is not None:
            dset.attrs['long_name'] = np.string_(long_name)

        if units is not None:
            dset.attrs['units'] = np.string_(units)

        if fill_value is not None:
            dset.attrs.create('_FillValue', data=fill_value)

        if valid_min is not None:
            dset.attrs.create('valid_min', data=valid_min)

        if valid_max is not None:
            dset.attrs.create('valid_max', data=valid_max)

    del gdal_ds


if __name__ == "__main__":
    cfg = runconfig.load('GCOV')
    h5_prep.run(cfg, 'GCOV')
    run(cfg)
