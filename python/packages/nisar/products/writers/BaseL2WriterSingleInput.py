from osgeo import gdal
import tempfile
import numpy as np
import warnings
import journal
import os

import isce3
from nisar.products.writers import BaseWriterSingleInput
from nisar.workflows.h5_prep import set_get_geo_info
from isce3.core.types import truncate_mantissa
from nisar.products.readers.orbit import load_orbit_from_xml


def save_dataset(ds_filename, h5py_obj, root_path,
                 yds, xds, ds_name,
                 compression_enabled=True,
                 compression_type='gzip',
                 compression_level=9,
                 mantissa_nbits=None,
                 standard_name=None,
                 long_name=None,
                 units=None,
                 fill_value=None,
                 valid_min=None,
                 valid_max=None,
                 compute_stats=True):
    '''
    Write a raster files as a set of HDF5 datasets

    Parameters
    ----------
    ds_filename : str
        Source raster file
    h5py_obj : h5py.File
        h5py object of destination HDF5
    root_path : str
        Path of output raster data
    yds : h5py.Dataset
        Y-axis dataset
    xds : h5py.Dataset
        X-axis dataset
    ds_name : str
        name of dataset to be added to root_path
    compression_enabled : bool, optional
        Enable/disable compression of output raster
        (only applicable for "file_format" equal to "HDF5")
    compression_type : str or None, optional
        Output data compression
        (only applicable for "file_format" equal to "HDF5")
    compression_level : int or None, optional
        Output data compression level
        (only applicable for "file_format" equal to "HDF5")
    mantissa_nbits: int or None, optional
        Number of mantissa bits for each real-part sample
        (only applicable for "file_format" equal to "HDF5")
    standard_name : string, optional
        HDF5 dataset standard name
    long_name : string, optional
        HDF5 dataset long name
    units : str, optional
        Value to populate the HDF5 dataset attribute "units"
    fill_value : float, optional
        Value to populate the HDF5 dataset attribute "_FillValue".
        Defaults to "nan" or "(nan+nanj)" if the raster layer
        is real- or complex-valued, respectively. If the layer data
        type is integer, the attribute "_FillValue"
        will not be populated as an attribute of the HDF5 dataset.
    valid_min : float, optional
        Value to populate the HDF5 dataset attribute "valid_min"
    valid_max : float, optional
        Value to populate the HDF5 dataset attribute "valid_max"
    '''
    if not os.path.isfile(ds_filename):
        return

    stats_real_imag_vector = None
    stats_vector = None
    if compute_stats:
        raster = isce3.io.Raster(ds_filename)

        if (raster.datatype() == gdal.GDT_CFloat32 or
                raster.datatype() == gdal.GDT_CFloat64):
            stats_real_imag_vector = \
                isce3.math.compute_raster_stats_real_imag(raster)
        elif raster.datatype() == gdal.GDT_Float64:
            stats_vector = isce3.math.compute_raster_stats_float64(raster)
        else:
            stats_vector = isce3.math.compute_raster_stats_float32(raster)

    create_dataset_kwargs = {}

    if compression_enabled and compression_type is not None:
        create_dataset_kwargs['compression'] = compression_type

    if compression_enabled and compression_level is not None:
        create_dataset_kwargs['compression_opts'] = \
            compression_level

    gdal_ds = gdal.Open(ds_filename, gdal.GA_ReadOnly)
    nbands = gdal_ds.RasterCount
    for band in range(nbands):
        data = gdal_ds.GetRasterBand(band+1).ReadAsArray()

        if mantissa_nbits is not None:
            truncate_mantissa(data, mantissa_nbits)

        # If we are saving multiple layers, `ds_name` is a list
        # Otherwise, it may be a list of a single element or
        # a string
        if isinstance(ds_name, str):
            h5_ds = os.path.join(root_path, ds_name)
        else:
            h5_ds = os.path.join(root_path, ds_name[band])

        dset = h5py_obj.require_dataset(h5_ds, data=data,
                                        shape=data.shape,
                                        dtype=data.dtype,
                                        **create_dataset_kwargs)

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
        elif 'cfloat' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan + 1j * np.nan)
        elif 'float' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan)

        if stats_vector is not None:
            stats_obj = stats_vector[band]
            dset.attrs.create('min_value', data=stats_obj.min)
            dset.attrs.create('mean_value', data=stats_obj.mean)
            dset.attrs.create('max_value', data=stats_obj.max)
            dset.attrs.create('sample_standard_deviation',
                              data=stats_obj.sample_stddev)

        elif stats_real_imag_vector is not None:

            stats_obj = stats_real_imag_vector[band]
            dset.attrs.create('min_real_value', data=stats_obj.real.min)
            dset.attrs.create('mean_real_value', data=stats_obj.real.mean)
            dset.attrs.create('max_real_value', data=stats_obj.real.max)
            dset.attrs.create('sample_standard_deviation_real',
                              data=stats_obj.real.sample_stddev)

            dset.attrs.create('min_imag_value', data=stats_obj.imag.min)
            dset.attrs.create('mean_imag_value', data=stats_obj.imag.mean)
            dset.attrs.create('max_imag_value', data=stats_obj.imag.max)
            dset.attrs.create('sample_standard_deviation_imag',
                              data=stats_obj.imag.sample_stddev)

        if valid_min is not None:
            dset.attrs.create('valid_min', data=valid_min)

        if valid_max is not None:
            dset.attrs.create('valid_max', data=valid_max)


class BaseL2WriterSingleInput(BaseWriterSingleInput):
    """
    Base L2 writer class that can be used for NISAR L2 products
    """

    def __init__(self, runconfig, *args, **kwargs):

        super().__init__(runconfig, *args, **kwargs)

        # if provided, load an external orbit from the runconfig file;
        # othewise, load the orbit from the RSLC metadata
        self.orbit_file = \
            self.cfg["dynamic_ancillary_file_group"]['orbit_file']
        self.flag_external_orbit_file = self.orbit_file is not None

        if self.flag_external_orbit_file:
            self.orbit = load_orbit_from_xml(self.orbit_file)
        else:
            orbit_path = (f'{self.root_path}/'
                          f'{self.input_product_hdf5_group_type}'
                          '/metadata/orbit')
            self.orbit = isce3.core.load_orbit_from_h5_group(
                self.input_hdf5_obj[orbit_path])

    def populate_identification_l2_specific(self):
        """
        Populate L2 product specific parameters in the
        identification group
        """

        self.copy_from_input(
            'identification/zeroDopplerStartTime')

        self.copy_from_input(
            'identification/zeroDopplerEndTime')

        self.set_value(
            'identification/productLevel',
            'L2')

        is_geocoded = True
        self.set_value(
            'identification/isGeocoded',
            is_geocoded)

        # TODO populate attribute `epsg`
        self.copy_from_input(
            'identification/boundingPolygon')

        self.set_value(
            'identification/listOfFrequencies',
            list(self.freq_pols_dict.keys()))

    def populate_calibration_information(self):

        # calibration parameters to be copied from the RSLC
        # common to all polarizations
        calibration_freq_parameter_list = ['commonDelay',
                                           'faradayRotation']

        # calibration parameters to be copied from the RSLC
        # specific to each polarization
        calibration_freq_pol_parameter_list = \
            ['differentialDelay',
             'differentialPhase',
             'scaleFactor',
             'scaleFactorSlope']

        for frequency in self.freq_pols_dict.keys():

            for parameter in calibration_freq_parameter_list:
                cal_freq_path = (
                    '{PRODUCT}/metadata/calibrationInformation/'
                    f'frequency{frequency}')

                self.copy_from_input(f'{cal_freq_path}/{parameter}',
                                     default=np.nan)

            # The following parameters are available for all
            # quad pols in the lexicographic base
            # regardless of listOfPolarizations
            for pol in ['HH', 'HV', 'VH', 'VV']:
                for parameter in calibration_freq_pol_parameter_list:
                    self.copy_from_input(f'{cal_freq_path}/{pol}/{parameter}',
                                         default=np.nan)

        # geocode crosstalk parameter LUTs
        crosstalk_parameters = ['txHorizontalCrosspol',
                                'txVerticalCrosspol',
                                'rxHorizontalCrosspol',
                                'rxVerticalCrosspol']
        self.geocode_lut(
            '{PRODUCT}/metadata/calibrationInformation/crosstalk',
            frequency=list(self.freq_pols_dict.keys())[0],
            output_ds_name_list=crosstalk_parameters,
            skip_if_not_present=True)

        luts_list = ['elevationAntennaPattern', 'nes0']

        for lut in luts_list:

            # geocode frequency dependent LUTs
            for frequency, pol_list in self.freq_pols_dict.items():

                # The path below is only valid for RSLC products
                # with product specification version 1.1.0 or above
                success = self.geocode_lut(
                    '{PRODUCT}/metadata/calibrationInformation/'
                    f'frequency{frequency}/{lut}',
                    frequency=frequency,
                    output_ds_name_list=pol_list,
                    skip_if_not_present=True)

                if not success:
                    break

            if success:
                continue

            # The code below handles RSLC products with
            # product specification version prior to 1.1.0
            for frequency, pol_list in self.freq_pols_dict.items():
                for pol in pol_list:

                    input_ds_name_list = [f'frequency{frequency}/{pol}/{lut}']

                    self.geocode_lut(
                        output_h5_group=('{PRODUCT}/metadata/'
                                         'calibrationInformation'
                                         f'/frequency{frequency}/{lut}'),
                        input_h5_group=('{PRODUCT}/metadata/'
                                        'calibrationInformation'),
                        frequency=list(self.freq_pols_dict.keys())[0],
                        input_ds_name_list=input_ds_name_list,
                        output_ds_name_list=pol,
                        skip_if_not_present=True)

    def populate_orbit(self):

        # TODO: add capability to store an external orbit file
        if self.flag_external_orbit_file:
            warnings.warn(
                'An external orbit file was used to create this L2 product.'
                ' However, capability of storing the external orbit data'
                ' into an L2 product is not yet implemented. Therefore,'
                ' the orbit data from the RSLC will be copied from the'
                ' input RSLC.')

        # RSLC products with product specification version prior to v1.1.0 may
        # include the H5 group "acceleration", that should not be copied to L2
        # products
        excludes_list = ['acceleration']

        # copy orbit information group
        self._copy_group_from_input('{PRODUCT}/metadata/orbit',
                                    excludes=excludes_list)

    def populate_attitude(self):
        # RSLC products with product specification version prior to v1.1.0 may
        # include the H5 group "angularVelocity", that should not be copied to
        # L2 products
        excludes_list = ['angularVelocity']

        # copy attitude information group
        self._copy_group_from_input('{PRODUCT}/metadata/attitude',
                                    excludes=excludes_list)

    def populate_processing_information_l2_common(self):
        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'softwareVersion',
            isce3.__version__)

        # populate processing information inputs parameters
        inputs_group = \
            '{PRODUCT}/metadata/processingInformation/inputs'

        self.set_value(
            f'{inputs_group}/l1SlcGranules',
            [self.input_file])

        orbit_file = self.cfg[
            'dynamic_ancillary_file_group']['orbit_file']
        if orbit_file is None:
            orbit_file = '(NOT SPECIFIED)'
        self.set_value(
            f'{inputs_group}/orbitFiles',
            [orbit_file])

        # `run_config_path` can be either a file name or a string
        # representing the contents of the runconfig file (identified
        # by the presence of a "/n" in the `run_config_path`)
        if '\n' not in self.runconfig.args.run_config_path:
            self.set_value(
                f'{inputs_group}/configFiles',
                [self.runconfig.args.run_config_path])

        self.copy_from_runconfig(
            f'{inputs_group}/demSource',
            'dynamic_ancillary_file_group/dem_file_description',
            default='(NOT SPECIFIED)')

        # geocode dopplerCentroid LUT
        ds_name_list = ['dopplerCentroid']
        for frequency in self.freq_pols_dict.keys():

            self.geocode_lut(
                '{PRODUCT}/metadata/processingInformation/parameters/'
                f'frequency{frequency}',
                frequency=frequency,
                output_ds_name_list=ds_name_list,
                skip_if_not_present=True)

    def geocode_lut(self, output_h5_group, input_h5_group=None,
                    frequency='A', output_ds_name_list=None,
                    input_ds_name_list=None,
                    skip_if_not_present=False):
        """
        Geocode a look-up table (LUT) from the input product in
        radar coordinates to the output product in map coordinates

        Parameters
        ----------
        output_h5_group: str
            Path to the output HDF5 LUT group
        input_h5_group: str, optional
            Path to the input HDF5 LUT group. If not provided, the
            same path of the output dataset `output_h5_group` will
            be used
        frequency: str, optional
            Frequency sub-band
        output_ds_name_list: str, list
            List of LUT datasets to geocode
        input_ds_name_list: list
            List of LUT datasets to geocode. If not provided, the
            same list as the `output_ds_name_list` will be used
        skip_if_not_present: bool, optional
            Flag to prevent the execution to stop if the dataset
            is not present from input

        Returns
        -------
        success: bool
           Flag that indicates if the geocoding the LUT group was successful
        """
        if not output_ds_name_list:
            return False

        if input_h5_group is None:
            input_h5_group = output_h5_group

        if input_ds_name_list is None and isinstance(output_ds_name_list, str):
            input_ds_name_list = [output_ds_name_list]
        elif input_ds_name_list is None:
            input_ds_name_list = output_ds_name_list

        input_h5_group_path = self.root_path + '/' + input_h5_group
        input_h5_group_path = \
            input_h5_group_path.replace(
                '{PRODUCT}', self.input_product_hdf5_group_type)

        error_channel = journal.error('geocode_lut')

        # check if group exists within the input product
        if input_h5_group_path not in self.input_hdf5_obj:
            not_found_msg = ('Metadata entry not found in the input'
                             ' product: ' + input_h5_group_path)
            if skip_if_not_present:
                warnings.warn(not_found_msg)
                return False
            else:
                error_channel.log(not_found_msg)
                raise KeyError(not_found_msg)

        output_h5_group_path = self.root_path + '/' + output_h5_group
        output_h5_group_path = \
            output_h5_group_path.replace('{PRODUCT}', self.product_type)

        is_calibration_information_group = \
            'calibrationInformation' in output_h5_group
        is_processing_information_group = \
            'processingInformation' in output_h5_group

        if (is_calibration_information_group and
                is_processing_information_group):
            error_msg = ('Malformed input product group:'
                         f' {output_h5_group}. Should'
                         ' contain either "calibrationInformation"'
                         ' or "processingInformation"')
            error_channel.log(error_msg)
            raise ValueError(error_msg)
        if is_calibration_information_group:
            metadata_group = 'calibrationInformation'
        elif is_processing_information_group:
            metadata_group = 'processingInformation'
        else:
            error_msg = f'Could not determine LUT group for {output_h5_group}'
            error_channel.log(error_msg)
            raise NotImplementedError(error_msg)

        return self.geocode_metadata_group(
            frequency,
            input_ds_name_list,
            output_ds_name_list,
            metadata_group,
            input_h5_group_path,
            output_h5_group_path,
            skip_if_not_present)

    def geocode_metadata_group(self,
                               frequency,
                               input_ds_name_list,
                               output_ds_name_list,
                               metadata_group,
                               input_h5_group_path,
                               output_h5_group_path,
                               skip_if_not_present):

        error_channel = journal.error('geocode_metadata_group')

        scratch_path = self.cfg['product_path_group']['scratch_path']

        if metadata_group == 'calibrationInformation':
            metadata_geogrid = self.cfg['processing'][
                'calibration_information']['geogrid']
        elif metadata_group == 'processingInformation':
            metadata_geogrid = self.cfg['processing'][
                'processing_information']['geogrid']
        else:
            error_msg = f'Invalid metadata group {metadata_group}'
            error_channel.log(error_msg)
            raise NotImplementedError(error_msg)

        dem_file = self.cfg['dynamic_ancillary_file_group']['dem_file']

        # unpack geo2rdr parameters
        geo2rdr_dict = self.cfg['processing']['geo2rdr']
        threshold = geo2rdr_dict['threshold']
        maxiter = geo2rdr_dict['maxiter']

        # init parameters shared between frequencyA and frequencyB sub-bands
        dem_raster = isce3.io.Raster(dem_file)
        zero_doppler = isce3.core.LUT2d()

        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid

        # do not apply any exponentiation to the samples to geocode
        exponent = 1

        geocode_mode = isce3.geocode.GeocodeOutputMode.INTERP

        radar_grid_slc = self.input_product_obj.getRadarGrid(frequency)

        zero_doppler_path = f'{input_h5_group_path}/zeroDopplerTime'
        try:
            zero_doppler_time_array = self.input_hdf5_obj[zero_doppler_path]
        except KeyError:
            not_found_msg = ('Metadata entry not found in the input'
                             ' product: ' + zero_doppler_path)
            if skip_if_not_present:
                warnings.warn(not_found_msg)
                return False
            else:
                error_channel.log(not_found_msg)
                raise KeyError(not_found_msg)

        slant_range_path = f'{input_h5_group_path}/slantRange'
        try:
            slant_range_array = self.input_hdf5_obj[slant_range_path]
        except KeyError:
            not_found_msg = ('Metadata entry not found in the input'
                             ' product: ' + slant_range_path)
            if skip_if_not_present:
                warnings.warn(not_found_msg)
                return False
            else:
                error_channel.log(not_found_msg)
                raise KeyError(not_found_msg)

        lines = zero_doppler_time_array.size
        samples = slant_range_array.size

        time_spacing = np.average(
            zero_doppler_time_array[1:-1] - zero_doppler_time_array[0:-2])
        range_spacing = np.average(
            slant_range_array[1:-1] - slant_range_array[0:-2])

        if time_spacing <= 0:
            error_msg = ('Invalid zero-Doppler time array under'
                         f' {zero_doppler_path}:'
                         f' {zero_doppler_time_array[()]}')
            error_channel.log(error_msg)
            raise RuntimeError(error_msg)

        if range_spacing <= 0:
            error_msg = ('Invalid range spacing array under'
                         f' {slant_range_path}: {slant_range_array[()]}')
            error_channel.log(error_msg)
            raise RuntimeError(error_msg)

        radar_grid = isce3.product.RadarGridParameters(
                zero_doppler_time_array[0],
                radar_grid_slc.wavelength,
                time_spacing,
                slant_range_array[0],
                range_spacing,
                radar_grid_slc.lookside,
                lines, samples, self.orbit.reference_epoch)

        # construct input rasters
        input_raster_list = []

        '''
        Create list of input Raster objects. The list will be used to
        create the input VRT file and raster (input_raster_obj).
        input_ds_name_list defines the target HDF5 dataset that will contain
        the geocoded metadata
        '''
        for var in input_ds_name_list:
            raster_ref = (f'HDF5:"{self.input_file}":/'
                          f'{input_h5_group_path}/{var}')
            temp_raster = isce3.io.Raster(raster_ref)
            input_raster_list.append(temp_raster)

        if len(input_ds_name_list) == 1:
            input_ds_name_list = input_ds_name_list[0]

        # create a temporary file that will point to the input layers
        input_temp = tempfile.NamedTemporaryFile(
            dir=scratch_path, suffix='.vrt')
        input_raster_obj = isce3.io.Raster(
            input_temp.name, raster_list=input_raster_list)

        # init Geocode object depending on raster type
        if input_raster_obj.datatype() == gdal.GDT_Float32:
            geo = isce3.geocode.GeocodeFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_Float64:
            geo = isce3.geocode.GeocodeFloat64()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat32:
            geo = isce3.geocode.GeocodeCFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat64:
            geo = isce3.geocode.GeocodeCFloat64()
        else:
            err_str = 'Unsupported raster type for geocoding'
            error_channel.log(err_str)
            raise NotImplementedError(err_str)

        # init geocode members
        geo.orbit = self.orbit
        geo.ellipsoid = ellipsoid
        geo.doppler = zero_doppler
        geo.threshold_geo2rdr = threshold
        geo.numiter_geo2rdr = maxiter

        geo.geogrid(metadata_geogrid.start_x, 
                    metadata_geogrid.start_y,
                    metadata_geogrid.spacing_x, 
                    metadata_geogrid.spacing_y,
                    metadata_geogrid.width, 
                    metadata_geogrid.length, 
                    metadata_geogrid.epsg)

        # set paths temporary files
        input_temp = tempfile.NamedTemporaryFile(
            dir=scratch_path, suffix='.vrt')
        input_raster_obj = isce3.io.Raster(
            input_temp.name, raster_list=input_raster_list)

        # create output raster
        temp_output = tempfile.NamedTemporaryFile(
            dir=scratch_path, suffix='.tif')

        dtype = input_raster_obj.datatype()

        output_raster_obj = isce3.io.Raster(
            temp_output.name, metadata_geogrid.width, metadata_geogrid.length,
            input_raster_obj.num_bands, dtype, 'GTiff')

        # geocode rasters
        geo.geocode(radar_grid=radar_grid,
                    input_raster=input_raster_obj,
                    output_raster=output_raster_obj,
                    output_mode=geocode_mode,
                    dem_raster=dem_raster,
                    exponent=exponent)

        output_raster_obj.close_dataset()
        del output_raster_obj

        x_coord_path = f'{output_h5_group_path}/xCoordinates'
        y_coord_path = f'{output_h5_group_path}/yCoordinates'

        if (x_coord_path not in self.output_hdf5_obj or
                y_coord_path not in self.output_hdf5_obj):
            yds, xds, *_ = set_get_geo_info(
                self.output_hdf5_obj, output_h5_group_path, metadata_geogrid,
                flag_save_coordinate_spacing=False)
        else:
            xds = self.output_hdf5_obj[x_coord_path]
            yds = self.output_hdf5_obj[y_coord_path]

        save_dataset(temp_output.name, self.output_hdf5_obj,
                     output_h5_group_path,
                     yds, xds, output_ds_name_list)

        return True
