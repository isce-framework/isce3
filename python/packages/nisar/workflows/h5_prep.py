"""
Suite of functions to prepare HDF5 for
GSLC, GCOV, GUNW, GOFF, RIFG, ROFF, and RUNW
"""

import os

import h5py
import journal
import numpy as np
from osgeo import osr

import isce3
from nisar.h5 import cp_h5_meta_data
from nisar.products.readers import SLC
from isce3.core.types import complex32, to_complex32
from nisar.workflows.helpers import get_cfg_freq_pols


def get_dataset_output_options(cfg: dict):
    '''
    Process chunking and compression options for GSLC from runconfing and
    return as kwargs dict that can be passed to h5py.Dataset.create_dataset

    Parameters
    ----------
    cfg: dict
        Runconfig containing output options

    Returns
    -------
    gslc_output_options: dict
        Dict containing chunking and compression options that can be passed
        to h5py.Dataset.create_dataset as kwargs
    '''
    # start with empty dict and populate as needed
    gslc_output_options = {}

    output_cfg = cfg['output']

    gslc_output_options['chunks'] = tuple(output_cfg['chunk_size'])

    if output_cfg['compression_enabled']:
        gslc_output_options['compression'] = 'gzip'
        gslc_output_options['compression_opts'] = \
            output_cfg['compression_level']
        gslc_output_options['shuffle'] = output_cfg['shuffle']

    return gslc_output_options


def get_complex_output_dtype(cfg: dict, dst_h5: h5py.File):
    '''
    Get the dtype for the complex output and the corresponding NaN fill value.
    
    The dtype is specified in `cfg`. `fill_value` will have a dtype of `ctype`.
    This custom ctype will be added as a new object in the root of `dst_h5` file,
    to teach tools like GDAL/netCDF how to read datasets with these custom dtypes.

    Parameters
    ----------
    cfg: dict
        Runconfig containing output options
    dst_h5: h5py.File
        Where datatype is to be committed

    Returns
    -------
    ctype: Union[h5py.h5t.TypeCompoundID, numpy.dtype]
        Datatype to be passed to h5py.Dataset.create_dataset
    fill_value: complex32 or complex64
        Value to initialize raster with
    '''
    output_type = cfg['output']['data_type']
    fill_value = np.nan * (1 + 1j)
    if output_type == 'complex32':
        # Creates a new object in the HDF5 file that teaches tools like
        # GDAL/netCDF about this custom datatype.
        h5_type = h5py.h5t.py_create(complex32)
        h5_type.commit(dst_h5['/'].id, np.string_('complex32'))
        ctype = complex32
        fill_value = to_complex32(np.array([fill_value]))[0]
    else:
        # output_type is 'complex64' or 'complex64_zero_mantissa'.
        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(dst_h5['/'].id, np.string_('complex64'))
        # Cast fill value as np.complex64.
        fill_value = np.complex64(fill_value)

    return ctype, fill_value


def get_products_and_paths(cfg: dict) -> (dict, dict):
    '''
    Get sub-product type dict and sub-product HDF5 paths
    '''
    output_path = cfg['product_path_group']['sas_output_file']
    scratch = cfg['product_path_group']['scratch_path']
    product_type = cfg['primary_executable']['product_type']

    # dict keying product type with list with possible product type(s)
    insar_products = ['ROFF', 'GOFF', 'RIFG', 'RUNW', 'GUNW']
    product_dict = {'ROFF_RIFG_RUNW_GOFF_GUNW': insar_products,
                    'RIFG_RUNW_GUNW': insar_products[2:],
                    'ROFF_GOFF': insar_products[0:2],
                    'GUNW': insar_products[2:],
                    'GOFF': insar_products[0:2],
                    'RUNW': ['RIFG', 'RUNW'],
                    'RIFG': ['RIFG'],
                    'ROFF': ["ROFF"],
                    'GCOV': ['GCOV'],
                    'GSLC': ['GSLC'],
                    }

    # dict keying product type to dict of product type key(s) to output(s)
    # following lambda creates subproduct specific output path
    insar_path = lambda out_path, product: \
        os.path.join(os.path.dirname(out_path),
                     product + '_' + os.path.basename(out_path))
    h5_paths = {'ROFF_RIFG_RUNW_GOFF_GUNW': dict(zip(insar_products,
                                                     [insar_path(output_path, product) for product
                                                     in insar_products])),
                'RIFG_RUNW_GUNW': dict(zip(insar_products[2:],
                                           [insar_path(output_path, product) for product
                                           in insar_products[2:]])),
                'ROFF_GOFF': dict(zip(insar_products[0:2],
                                      [insar_path(output_path, product) for product
                                       in insar_products[0:2]])),
                'GUNW': {'RIFG': f'{scratch}/RIFG.h5',
                         'RUNW': f'{scratch}/RUNW.h5',
                         'GUNW': output_path},
                'GOFF': {'ROFF': f'{scratch}/ROFF.h5',
                         'GOFF': output_path},
                'RUNW': {'RIFG': f'{scratch}/RIFG.h5', 'RUNW': output_path},
                'RIFG': {'RIFG': output_path},
                'ROFF': {'ROFF': output_path},
                'GCOV': {'GCOV': output_path},
                'GSLC': {'GSLC': output_path},
                'RUNW_STANDALONE': {'RUNW': output_path},
                'GUNW_STANDALONE': {'GUNW': output_path}}

    return product_dict[product_type], h5_paths[product_type]


def run(cfg: dict) -> dict:
    '''
    Copy metadata from src hdf5 and prepare datasets
    Returns dict of output path(s); used for InSAR workflow
    '''
    info_channel = journal.info("h5_prep.run")
    info_channel.log('preparing HDF5')

    product_dict, h5_paths = get_products_and_paths(cfg)
    for sub_prod_type in product_dict:
        out_path = h5_paths[sub_prod_type]
        cp_geocode_meta(cfg, out_path, sub_prod_type)
        prep_ds(cfg, out_path, sub_prod_type)

    info_channel.log('successfully ran h5_prep')

    return h5_paths


def cp_geocode_meta(cfg, output_hdf5, dst):
    '''
    Copy shared data from source HDF5 to GSLC, GCOV, INSAR
    HDF5 destinations
    Parameters:
    -----------
    cfg : dict
        Run configuration
    dst : str or list
        Name of destination node where data is to be copied
    '''

    # Check if InSAR
    is_insar = dst in ['GUNW', 'GOFF', 'RUNW', 'ROFF', 'RIFG']

    # unpack info
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    if is_insar:
        input_hdf5 = cfg['input_file_group']['reference_rslc_file']
        secondary_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    else:
        input_hdf5 = cfg['input_file_group']['input_file_path']

    rtc_algorithm = ''
    if dst == "GCOV":
        dem_interp_method = cfg['processing']['dem_interpolation_method']
        geocoding_algorithm = cfg['processing']['geocode']['algorithm_type']
        rtc_algorithm = cfg['processing']['rtc']['algorithm_type']
    elif dst == "GSLC":
        dem_interp_method = 'biquintic'
        geocoding_algorithm = 'sinc'
    elif dst == "GUNW":
        dem_interp_method = 'biquintic'
        geocoding_algorithm = cfg["processing"]["geocode"]["interp_method"]
    else:
        dem_interp_method = ''
        geocoding_algorithm = ''

    # Remove existing HDF5 and start from scratch
    try:
        os.remove(output_hdf5)
    except FileNotFoundError:
        pass

    # Open reference slc
    ref_slc = SLC(hdf5file=input_hdf5)

    # prelim setup
    common_path = 'science/LSAR'
    src_meta_path = ref_slc.MetadataPath
    dst_meta_path = f'{common_path}/{dst}/metadata'

    with h5py.File(input_hdf5, 'r', libver='latest', swmr=True) as src_h5, \
            h5py.File(output_hdf5, 'w', libver='latest', swmr=True) as dst_h5:
        dst_h5.attrs['Conventions'] = np.string_("CF-1.7")

        # Copy of identification group
        ident_path = f'{common_path}/identification'
        ident_excludes = ['productType', 'productVersion',
                          'productSpecificationVersion', 'granuleId']
        if is_insar:
            ident_excludes += ['listOfFrequencies', 'zeroDopplerStartTime',
                               'zeroDopplerEndTime']
        cp_h5_meta_data(src_h5, dst_h5, ident_path,
                        excludes=ident_excludes)

        # If insar, create reference/secondary zeroDopplerStartEndTime
        if is_insar:
            # Open secondary hdf5 to copy information
            with h5py.File(secondary_hdf5, 'r', libver='latest', swmr=True) as sec_src_h5:
                src_dataset = ['zeroDopplerStartTime', 'zeroDopplerEndTime']
                dst_dataset = ['referenceZeroDopplerStartTime', 'referenceZeroDopplerEndTime']
                for src_data, dst_data in zip(src_dataset, dst_dataset):
                    cp_h5_meta_data(src_h5, dst_h5, f'{ident_path}/{src_data}',
                                    f'{ident_path}/{dst_data}')
                    dst_data = dst_data.replace('reference', 'secondary')
                    cp_h5_meta_data(sec_src_h5, dst_h5, f'{ident_path}/{src_data}',
                                    f'{ident_path}/{dst_data}')

        # Delete flag isGeocoded if exist, and assign it again
        ident = dst_h5[ident_path]
        if 'isGeocoded' in ident:
            del ident['isGeocoded']
        is_geocoded = dst in ['GCOV', 'GSLC', 'GUNW', 'GOFF']
        dst_h5[ident_path].create_dataset('isGeocoded',
                                          data=np.string_(str(is_geocoded)))
        desc = "Flag to indicate radar geometry or geocoded product"
        dst_h5[ident_path].attrs["description"] = np.string_(desc)

        # Assign productType
        dst_h5[f'{ident_path}/productType'] = np.string_(dst)

        # Assign product version
        dst_h5[f'{ident_path}/productVersion'] = \
            np.string_(cfg['primary_executable']['product_version'])

        # Assign product specification version
        dst_h5[f'{ident_path}/productSpecificationVersion'] = \
            np.string_('0.9.0')

        # Assign granule ID
        dst_h5[f'{ident_path}/granuleId'] = \
            np.string_(cfg['primary_executable']['partial_granule_id'])

        # copy orbit information group
        cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/orbit',
                        f'{dst_meta_path}/orbit')

        # copy attitude information group
        cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/attitude',
                        f'{dst_meta_path}/attitude')

        # copy processingInformation/algorithms group (common across products)
        if dst in ['ROFF', 'GOFF']:
            dst_proc_path = f'{dst_meta_path}/processingInformation/algorithms'
            excludes =['SWSTCorrection', 'azimuthCompression', 'azimuthPresumming',
                       'driftCompensator', 'elevationAntennaPatternCorrection',
                       'internalCalibration', 'rangeCellMigration',
                       'rangeCompression', 'rangeDependentGainCorrection',
                       'rangeReferenceFunctionGenerator', 'rangeSpreadingLossCorrection',
                       'secondaryRangeCompression']
            cp_h5_meta_data(src_h5, dst_h5,
                            f'{src_meta_path}/processingInformation/algorithms',
                            dst_proc_path, excludes=excludes)
            dst_h5[dst_proc_path].create_dataset('crossCorrelation',
                                                 data=np.string_('Ampcor'))
            descr = "Cross-correlation algorithm"
            dst_h5[dst_proc_path].attrs["description"] = np.string_(descr)
        else:
           cp_h5_meta_data(src_h5, dst_h5,
                           f'{src_meta_path}/processingInformation/algorithms',
                           f'{dst_meta_path}/processingInformation/algorithms')
        if dst in ['RIFG', 'RUNW', 'ROFF']:
            yds = dst_h5.create_dataset(f'{dst_meta_path}/geolocationGrid/zeroDopplerTime',
                   data = src_h5[f'{src_meta_path}/geolocationGrid/zeroDopplerTime'])
            xds = dst_h5.create_dataset(f'{dst_meta_path}/geolocationGrid/slantRange',
                   data = src_h5[f'{src_meta_path}/geolocationGrid/slantRange'])
            cp_h5_meta_data(src_h5, dst_h5,
                            f'{src_meta_path}/geolocationGrid',
                            f'{dst_meta_path}/geolocationGrid',
                            excludes=['zeroDopplerTime', 'slantRange'],
                            attach_scales_list=[yds, xds])
        if dst in ["GCOV", "GSLC", "GUNW"]:
            # Geocoding algorithm
            algorithms_ds = f'{dst_meta_path}/processingInformation/algorithms/geocoding'
            dset = dst_h5.require_dataset(algorithms_ds, (), "S27",
                                   data=np.string_(geocoding_algorithm))
            desc = "Geocoding algorithm"
            dset.attrs["description"] = np.string_(desc)

            # DEM interpolation method
            algorithms_ds = \
                f'{dst_meta_path}/processingInformation/algorithms/demInterpolation'
            dset = dst_h5.require_dataset(algorithms_ds, (), "S27",
                                          data=np.string_(dem_interp_method))
            desc = "DEM interpolation method"
            dset.attrs["description"] = np.string_(desc)

        if dst in ["GCOV"]:
            # RTC algorithm
            algorithms_ds = \
                f'{dst_meta_path}/processingInformation/algorithms/radiometricTerrainCorrection'
            dset = dst_h5.require_dataset(algorithms_ds, (), "S27",
                                          data=np.string_(rtc_algorithm))
            desc = "Radiometric terrain correction (RTC) algorithm"
            dset.attrs["description"] = np.string_(desc)

        # copy processingInformation/inputs group
        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_meta_path}/processingInformation/inputs',
                        f'{dst_meta_path}/processingInformation/inputs',
                        excludes=['l0bGranules'])
        # Create l1SlcGranules
        inputs = [input_hdf5]
        if is_insar:
            inputs.append(secondary_hdf5)
        input_grp = dst_h5[f'{dst_meta_path}/processingInformation/inputs']
        dset = input_grp.create_dataset("l1SlcGranules",
                                        data=np.string_(inputs))
        desc = "List of input L1 RSLC products used"
        dset.attrs["description"] = np.string_(desc)
        dset.attrs["long_name"] = np.string_("list of L1 RSLC products")

        # Copy processingInformation/parameters
        if dst in ['GUNW', 'GOFF']:
            exclude_args = ['frequencyA', 'frequencyB', 'azimuthChirpWeighting',
                            'effectiveVelocity', 'rangeChirpWeighting',
                            'slantRange', 'zeroDopplerTime']
        elif dst in ['RUNW', 'RIFG', 'ROFF']:
            exclude_args = ['frequencyA', 'frequencyB',
                            'azimuthChirpWeighting',
                            'effectiveVelocity', 'rangeChirpWeighting']
        else:
            exclude_args = ['nes0', 'elevationAntennaPattern']

        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_meta_path}/processingInformation/parameters',
                        f'{dst_meta_path}/processingInformation/parameters',
                        excludes=exclude_args)
        # Copy calibrationInformation group
        exclude_args = []
        if is_insar:
            exclude_args = ['nes0', 'elevationAntennaPattern']
        for freq in freq_pols.keys():
            if dst in ['ROFF', 'GOFF']:
                cal_path = f'{dst_meta_path}/calibrationInformation'
                dst_h5[f'{dst_meta_path}'].require_group('calibrationInformation')
                dst_h5[cal_path].create_group(f'frequency{freq}')
            else:
               pol_list = freq_pols[freq]
               if pol_list is None:
                  continue
               for pol in pol_list:
                   cp_h5_meta_data(src_h5, dst_h5,
                                   f'{src_meta_path}/calibrationInformation/frequency{freq}/{pol}',
                                   f'{dst_meta_path}/calibrationInformation/frequency{freq}/{pol}',
                                   excludes=exclude_args)

        # Copy product specifics
        if is_insar:
            copy_insar_meta(cfg, dst, src_h5, dst_h5, src_meta_path)
        else:
            copy_gslc_gcov_meta(ref_slc.SwathPath, dst, src_h5, dst_h5)
            if ref_slc.productType in ref_slc.SwathPath:
                # Regular case
                dst_path = ref_slc.SwathPath.replace(ref_slc.productType, dst)
            else:
                # ProductType is RSLC and SwathPath contains /SLC/
                dst_path = ref_slc.SwathPath.replace('SLC', dst)
            # Copy zeroDopplerTimeSpacing scalar (GCOV and GSLC)
            for freq in freq_pols.keys():
                frequency = f'frequency{freq}'
                dst_freq_path = dst_path.replace('swaths', 'grids')+f'/{frequency}'
                copy_zero_doppler_time_spacing(src_h5, ref_slc.SwathPath,
                                               dst_h5, dst_freq_path)
        src_h5.close()
        dst_h5.close()


def copy_zero_doppler_time_spacing(src_h5, swath_path, dst_h5, dst_path):
    az_spacing = src_h5[f'{swath_path}/zeroDopplerTimeSpacing'][()]
    descr = "Time interval in the along track direction for raster layers. " \
            "This is the same as the spacing between consecutive entries in " \
            "zeroDopplerTime array"
    _create_datasets(dst_h5[dst_path], [0], np.float32,
                     'zeroDopplerTimeSpacing',
                     descr=descr, units="seconds", data=az_spacing,
                     long_name="zero doppler time spacing")


def copy_gslc_gcov_meta(src_swath_path, dst, src_h5, dst_h5):
    '''
    Copy metadata info for GSLC GCOV workflows
    '''
    # prelim setup
    common_parent_path = 'science/LSAR'

    for freq in ['A', 'B']:
        ds_ref = f'{src_swath_path}/frequency{freq}'
        if ds_ref not in src_h5:
            continue

        cp_h5_meta_data(src_h5, dst_h5, ds_ref,
                        os.path.join(common_parent_path,
                                     f'{dst}/grids/frequency{freq}'),
                        excludes=['acquiredCenterFrequency',
                                  'acquiredAzimuthBandwidth',
                                  'acquiredRangeBandwidth',
                                  'nominalAcquisitionPRF', 'slantRange',
                                  'sceneCenterAlongTrackSpacing',
                                  'sceneCenterGroundRangeSpacing',
                                  'HH', 'HV', 'VH', 'VV', 'RH', 'RV',
                                  'validSamplesSubSwath1',
                                  'validSamplesSubSwath2',
                                  'validSamplesSubSwath3',
                                  'validSamplesSubSwath4',
                                  'listOfPolarizations'],
                        renames={'processedCenterFrequency': 'centerFrequency',
                                 'processedAzimuthBandwidth': 'azimuthBandwidth',
                                 'processedRangeBandwidth': 'rangeBandwidth'},
                        flag_overwrite=True)


def copy_insar_meta(cfg, dst, src_h5, dst_h5, src_meta_path):
    '''
    Copy metadata specific to INSAR workflow
    '''
    common_path = 'science/LSAR'
    dst_meta_path = f'{common_path}/{dst}/metadata'

    secondary_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # Open secondary SLC
    with h5py.File(secondary_hdf5, 'r', libver='latest',
                   swmr=True) as secondary_h5:
        dst_proc = f'{dst_meta_path}/processingInformation/parameters'
        src_proc = f'{src_meta_path}/processingInformation/parameters'

        # Create groups in processing Information
        dst_h5.create_group(f'{dst_proc}/common')
        dst_h5.create_group(f'{dst_proc}/reference')
        dst_h5.create_group(f'{dst_proc}/secondary')

        # Copy data for reference and secondary
        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_proc}/effectiveVelocity',
                        f'{dst_proc}/reference/effectiveVelocity')
        cp_h5_meta_data(secondary_h5, dst_h5,
                        f'{src_proc}/effectiveVelocity',
                        f'{dst_proc}/secondary/effectiveVelocity')
        for freq in freq_pols.keys():
            frequency = f'frequency{freq}'
            cp_h5_meta_data(src_h5, dst_h5,
                            f'{src_proc}/{frequency}',
                            f'{dst_proc}/reference/{frequency}')
            cp_h5_meta_data(secondary_h5, dst_h5,
                            f'{src_proc}/{frequency}',
                            f'{dst_proc}/secondary/{frequency}')

        # Copy secondary image slantRange and azimuth time (modify attributes)
        dst_grid_path =f'{dst_meta_path}/radarGrid'
        if dst in ['RUNW', 'RIFG', 'ROFF']:
            dst_grid_path =f'{dst_meta_path}/geolocationGrid'

        cp_h5_meta_data(secondary_h5, dst_h5,
                        f'{src_meta_path}/geolocationGrid/slantRange',
                        f'{dst_grid_path}/secondarySlantRange')
        cp_h5_meta_data(secondary_h5, dst_h5,
                       f'{src_meta_path}/geolocationGrid/zeroDopplerTime',
                       f'{dst_grid_path}/secondaryZeroDopplerAzimuthTime')

        # Update these attribute with a description
        descr = "Slant range of corresponding pixels in secondary image"
        dst_h5[f'{dst_grid_path}/secondarySlantRange'].attrs[
            "description"] = descr
        descr = "Zero Doppler azimuth time of corresponding pixel in secondary image"
        dst_h5[f'{dst_grid_path}/secondaryZeroDopplerAzimuthTime'].attrs[
            "description"] = descr


def prep_ds(cfg, output_hdf5, dst):
    '''
    Prepare datasets for GSLC and GCOV
    '''
    # unpack
    with h5py.File(output_hdf5, 'a', libver='latest', swmr=True) as dst_h5:
        # Fork the dataset preparation for GSLC/GCOV and GUNW
        if dst in ['GSLC', 'GCOV']:
            prep_gslc_dataset(cfg, dst, dst_h5)
        else:
            prep_ds_insar(cfg, dst, dst_h5)

def prep_gslc_dataset(cfg, dst, dst_h5):
    '''
    Prepare datasets for GSLC and GCOV
    '''
    # unpack info
    common_parent_path = 'science/LSAR'
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    gslc_output_options = {}
    # if GSLC, populate output dict with h5py.Group.create_dataset kwargs

    # Get compression and chunking options
    gslc_output_options = get_dataset_output_options(cfg)

    # Get complex data type and set fill value to kwargs
    ctype, fill_value = get_complex_output_dtype(cfg, dst_h5)
    gslc_output_options['fillvalue'] = fill_value

    # Create datasets in the ouput hdf5
    geogrids = cfg['processing']['geocode']['geogrids']
    for freq, pol_list in freq_pols.items():
        shape = (geogrids[freq].length, geogrids[freq].width)
        dst_parent_path = os.path.join(common_parent_path,
                                       f'{dst}/grids/frequency{freq}')

        yds, xds = set_get_geo_info(dst_h5, dst_parent_path, geogrids[freq])

        # create datasets for polarizations of current frequency
        for polarization in pol_list:
            dst_grp = dst_h5[dst_parent_path]
            long_name = f'geocoded single-look complex image {polarization}'
            descr = f'Geocoded SLC image ({polarization})'
            _create_datasets(dst_grp, shape, ctype, polarization,
                             descr=descr, units='', grids="projection",
                             long_name=long_name, yds=yds, xds=xds,
                             fill_value=fill_value, **gslc_output_options)

        _add_polarization_list(dst_h5, dst, common_parent_path, freq, pol_list)

def get_off_params(pcfg, param_name, is_roff=False, pattern=None,
                   get_min=False):

    cfg = pcfg['offsets_product'] if is_roff else pcfg['dense_offsets']
    if pattern is not None and is_roff:
        param = [cfg[key].get(param_name, None) for key
                 in cfg if key.startswith(pattern)]
    else:
        param = cfg.get(param_name, None)

    if get_min and isinstance(param, list):
       param = min(list(filter(None, param)), default=0)

    return param


def _create_datasets(dst_grp, shape, ctype, dataset_name,
                     chunks=(128, 128), descr=None, units=None,
                     grids=None, data=None, standard_name=None, long_name=None,
                     yds=None, xds=None, fill_value=None, **kwargs):
    '''
    wrapper around h5py.Group.create_dataset that adds nisar.workflows specific
    attributes to avoid boilerplate calls
    '''
    if len(shape) == 1:
        if ctype == np.string_:
            ds = dst_grp.create_dataset(dataset_name,
                                        data=np.string_("         "))
        else:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, data=data)
    else:
        # do not create chunked dataset if any chunk dimension is larger than
        # dataset or is GUNW (temporary fix for CUDA geocode insar's inability
        # to direct write to HDF5 with chunks)
        # details https://github-fn.jpl.nasa.gov/isce-3/isce/issues/813
        create_with_chunks = (chunks[0] < shape[0] and chunks[1] < shape[1]) \
            and ('GUNW' not in dst_grp.name)
        if create_with_chunks:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, shape=shape,
                                        chunks=chunks, **kwargs)
        else:
            # create dataset without chunks
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, shape=shape,
                                        **kwargs)

    ds.attrs['description'] = np.string_(descr)

    if units is not None:
        ds.attrs['units'] = np.string_(units)

    if grids is not None:
        ds.attrs['grid_mapping'] = np.string_(grids)

    if standard_name is not None:
        ds.attrs['standard_name'] = np.string_(standard_name)

    if long_name is not None:
        ds.attrs['long_name'] = np.string_(long_name)

    if yds is not None:
        ds.dims[0].attach_scale(yds)

    if xds is not None:
        ds.dims[1].attach_scale(xds)

    if fill_value is not None:
        ds.attrs["_FillValue"] = fill_value


def _add_polarization_list(dst_h5, dst, common_parent_path, frequency, pols):
    '''
    Add list of processed polarizations
    '''
    dataset_path = os.path.join(common_parent_path,
                                f'{dst}/grids/frequency{frequency}')

    if dst in ['RUNW', 'RIFG', 'ROFF']:
        dataset_path = os.path.join(common_parent_path,
                                    f'{dst}/swaths/frequency{frequency}')

    grp = dst_h5[dataset_path]
    name = "listOfPolarizations"
    pols_array = np.array(pols, dtype="S2")
    dset = grp.create_dataset(name, data=pols_array)
    desc = f"List of processed polarization layers with frequency {frequency}"
    dset.attrs["description"] = np.string_(desc)


def set_get_geo_info(hdf5_obj, root_ds, geo_grid, z_vect=None,
                     flag_cube=False, flag_save_coordinate_spacing=True):
    epsg_code = geo_grid.epsg

    dx = geo_grid.spacing_x
    x0 = geo_grid.start_x + 0.5 * dx
    xf = x0 + (geo_grid.width - 1) * dx
    x_vect = np.linspace(x0, xf, geo_grid.width, dtype=np.float64)

    dy = geo_grid.spacing_y
    y0 = geo_grid.start_y + 0.5 * dy
    yf = y0 + (geo_grid.length - 1) * dy
    y_vect = np.linspace(y0, yf, geo_grid.length, dtype=np.float64)

    if epsg_code == 4326:
        x_coord_units = "degree_east"
        y_coord_units = "degree_north"
        x_standard_name = "longitude"
        y_standard_name = "latitude"
    else:
        x_coord_units = "meters"
        y_coord_units = "meters"
        x_standard_name = "projection_x_coordinate"
        y_standard_name = "projection_y_coordinate"

    if not flag_cube and flag_save_coordinate_spacing:
        # xCoordinateSpacing
        descr = (f'Nominal spacing in {x_coord_units}'
                 ' between consecutive pixels')
        xds_spacing_name = os.path.join(root_ds, 'xCoordinateSpacing')
        if xds_spacing_name in hdf5_obj:
            del hdf5_obj[xds_spacing_name]
        xds_spacing = hdf5_obj.create_dataset(xds_spacing_name, data=dx)
        xds_spacing.attrs["description"] = np.string_(descr)
        xds_spacing.attrs["units"] = np.string_(x_coord_units)
        xds_spacing.attrs["long_name"] = np.string_("x coordinate spacing")

        # yCoordinateSpacing
        descr = (f'Nominal spacing in {y_coord_units}'
                 ' between consecutive lines')
        yds_spacing_name = os.path.join(root_ds, 'yCoordinateSpacing')
        if yds_spacing_name in hdf5_obj:
            del hdf5_obj[yds_spacing_name]
        yds_spacing = hdf5_obj.create_dataset(yds_spacing_name, data=dy)
        yds_spacing.attrs["description"] = np.string_(descr)
        yds_spacing.attrs["units"] = np.string_(y_coord_units)
        yds_spacing.attrs["long_name"] = np.string_("y coordinates spacing")

    # xCoordinates
    descr = 'X coordinates in specified projection'
    xds_name = os.path.join(root_ds, 'xCoordinates')
    if xds_name in hdf5_obj:
        del hdf5_obj[xds_name]
    xds = hdf5_obj.create_dataset(xds_name, data=x_vect)
    xds.attrs['standard_name'] = x_standard_name
    xds.attrs["description"] = np.string_(descr)
    xds.attrs["units"] = np.string_(x_coord_units)
    xds.attrs["long_name"] = np.string_("x coordinate")

    # yCoordinates
    descr = 'Y coordinates in specified projection'

    yds_name = os.path.join(root_ds, 'yCoordinates')
    if yds_name in hdf5_obj:
        del hdf5_obj[yds_name]
    yds = hdf5_obj.create_dataset(yds_name, data=y_vect)
    yds.attrs['standard_name'] = y_standard_name
    yds.attrs["description"] = np.string_(descr)
    yds.attrs["units"] = np.string_(y_coord_units)
    yds.attrs["long_name"] = np.string_("y coordinate")

    coordinates_list = [xds, yds]

    # zCoordinates
    if z_vect is not None:
        descr = ('Height values above WGS84 Ellipsoid corresponding to the'
                 ' radar grid')
        zds_name = os.path.join(root_ds, 'heightAboveEllipsoid')
        if zds_name in hdf5_obj:
            del hdf5_obj[zds_name]
        zds = hdf5_obj.create_dataset(zds_name, data=z_vect, dtype='f8')
        zds.attrs['standard_name'] = np.string_(
            "height_above_reference_ellipsoid")
        zds.attrs["description"] = np.string_(descr)
        zds.attrs['units'] = np.string_("meters")
        coordinates_list.append(zds)

    try:
        for _ds in coordinates_list:
            _ds.make_scale()
    except AttributeError:
        pass

    # Associate grid mapping with data - projection created later
    projection_ds_name = os.path.join(root_ds, "projection")

    # Create a new single int dataset for projections
    if projection_ds_name in hdf5_obj:
        del hdf5_obj[projection_ds_name]
    projds = hdf5_obj.create_dataset(projection_ds_name, (), dtype='i')
    projds[()] = epsg_code

    # Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)

    # Add projection description
    projds.attrs['description'] = np.string_('Product map grid projection: EPSG code, '
                                             'with additional projection information as HDF5 Attributes')

    # WGS84 ellipsoid
    projds.attrs['semi_major_axis'] = 6378137.0
    projds.attrs['inverse_flattening'] = 298.257223563
    projds.attrs['ellipsoid'] = np.string_("WGS84")

    # Additional fields
    projds.attrs['epsg_code'] = epsg_code

    # CF 1.7+ requires this attribute to be named "crs_wkt"
    # spatial_ref is old GDAL way. Using that for testing only.
    # For NISAR replace with "crs_wkt"
    projds.attrs['spatial_ref'] = np.string_(srs.ExportToWkt())

    # Here we have handcoded the attributes for the different cases
    # Recommended method is to use pyproj.CRS.to_cf() as shown above
    # To get complete set of attributes.

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg_code)

    projds.attrs['grid_mapping_name'] = sr.GetName()

    # Set up units
    # Geodetic latitude / longitude
    if epsg_code == 4326:
        # Set up grid mapping
        projds.attrs['longitude_of_prime_meridian'] = 0.0
        projds.attrs['latitude_of_projection_origin'] = sr.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN)
        projds.attrs['longitude_of_projection_origin'] = sr.GetProjParm(
            osr.SRS_PP_LONGITUDE_OF_ORIGIN)

    else:
        # UTM zones
        if ((epsg_code > 32600 and
             epsg_code < 32661) or
                (epsg_code > 32700 and
                 epsg_code < 32761)):
            # Set up grid mapping
            projds.attrs['utm_zone_number'] = epsg_code % 100

        # Polar Stereo North
        elif epsg_code == 3413:
            # Set up grid mapping
            projds.attrs['standard_parallel'] = 70.0
            projds.attrs['straight_vertical_longitude_from_pole'] = -45.0

        # Polar Stereo south
        elif epsg_code == 3031:
            # Set up grid mapping
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0

        # EASE 2 for soil moisture L3
        elif epsg_code == 6933:
            # Set up grid mapping
            projds.attrs['longitude_of_central_meridian'] = 0.0
            projds.attrs['standard_parallel'] = 30.0

        # Europe Equal Area for Deformation map (to be implemented in isce3)
        elif epsg_code == 3035:
            # Set up grid mapping
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0

        else:
            raise NotImplementedError(
                f'EPSG {epsg_code} waiting for implementation / not supported in ISCE3')

        # Setup common parameters
        xds.attrs['long_name'] = np.string_("x coordinate of projection")
        yds.attrs['long_name'] = np.string_("y coordinate of projection")

        projds.attrs['false_easting'] = sr.GetProjParm(osr.SRS_PP_FALSE_EASTING)
        projds.attrs['false_northing'] = sr.GetProjParm(
            osr.SRS_PP_FALSE_NORTHING)

        projds.attrs['latitude_of_projection_origin'] = sr.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN)
        projds.attrs['longitude_of_projection_origin'] = sr.GetProjParm(
            osr.SRS_PP_LONGITUDE_OF_ORIGIN)

    if z_vect is not None:
        return zds, yds, xds
    return yds, xds


def add_radar_grid_cubes_to_hdf5(hdf5_obj, cube_group_name, geogrid,
                                 heights, radar_grid, orbit,
                                 native_doppler, grid_doppler,
                                 threshold_geo2rdr=1e-8,
                                 numiter_geo2rdr=100, delta_range=1e-8):
    if cube_group_name not in hdf5_obj:
        cube_group = hdf5_obj.create_group(cube_group_name)
    else:
        cube_group = hdf5_obj[cube_group_name]

    cube_shape = [len(heights), geogrid.length, geogrid.width]

    zds, yds, xds = set_get_geo_info(hdf5_obj, cube_group_name, geogrid,
                                     z_vect=heights, flag_cube=True)

    # seconds since ref epoch
    ref_epoch = radar_grid.ref_epoch
    ref_epoch_str = ref_epoch.isoformat().replace('T', ' ')
    az_coord_units = f'seconds since {ref_epoch_str}'

    slant_range_raster = _get_raster_from_hdf5_ds(
        cube_group, 'slantRange', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='slant-range',
        descr='Slant range in meters',
        units='meter')
    azimuth_time_raster = _get_raster_from_hdf5_ds(
        cube_group, 'zeroDopplerAzimuthTime', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='zero-Doppler azimuth time',
        descr='Zero Doppler azimuth time in seconds',
        units=az_coord_units)
    incidence_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'incidenceAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='incidence angle',
        descr=('Incidence angle is defined as the angle between the LOS vector'
               ' and the normal to the ellipsoid at the target height'),
        units='degrees')
    los_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector X',
        descr='East component of unit vector of LOS from target to sensor',
        units='1')
    los_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector Y',
        descr='North component of unit vector of LOS from target to sensor',
        units='1')
    along_track_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector X',
        descr='East component of unit vector along ground track',
        units='1')
    along_track_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector Y',
        descr='North component of unit vector along ground track',
        units='1')
    elevation_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'elevationAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Elevation angle',
        descr=('Elevation angle is defined as the angle between the LOS vector'
               ' and the normal to the ellipsoid at the sensor'),
        units='degrees')
    ground_track_velocity_raster = _get_raster_from_hdf5_ds(
        cube_group, 'groundTrackVelocity', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Ground-track velocity',
        descr=('Absolute value of the platform velocity scaled at the target'
               ' height'),
        units='m/s')

    isce3.geometry.make_radar_grid_cubes(radar_grid,
                                         geogrid,
                                         heights,
                                         orbit,
                                         native_doppler,
                                         grid_doppler,
                                         slant_range_raster,
                                         azimuth_time_raster,
                                         incidence_angle_raster,
                                         los_unit_vector_x_raster,
                                         los_unit_vector_y_raster,
                                         along_track_unit_vector_x_raster,
                                         along_track_unit_vector_y_raster,
                                         elevation_angle_raster,
                                         ground_track_velocity_raster,
                                         threshold_geo2rdr,
                                         numiter_geo2rdr,
                                         delta_range,
                                         flag_set_output_rasters_geolocation=False)


def _get_raster_from_hdf5_ds(group, ds_name, dtype, shape,
                             zds=None, yds=None, xds=None, standard_name=None,
                             long_name=None, descr=None,
                             units=None, fill_value=None,
                             valid_min=None, valid_max=None):
    # remove dataset if it already exists
    if ds_name in group:
        del group[ds_name]

    # create dataset
    dset = group.create_dataset(ds_name, dtype=dtype, shape=shape)

    if zds is not None:
        dset.dims[0].attach_scale(zds)
    if yds is not None:
        dset.dims[1].attach_scale(yds)
    if xds is not None:
        dset.dims[2].attach_scale(xds)

    dset.attrs['grid_mapping'] = np.string_("projection")

    if standard_name is not None:
        dset.attrs['standard_name'] = np.string_(standard_name)

    if long_name is not None:
        dset.attrs['long_name'] = np.string_(long_name)

    if descr is not None:
        dset.attrs["description"] = np.string_(descr)

    if units is not None:
        dset.attrs['units'] = np.string_(units)

    if fill_value is not None:
        dset.attrs.create('_FillValue', data=fill_value)
    elif np.issubdtype(dtype, np.floating):
        dset.attrs.create('_FillValue', data=np.nan)
    elif np.issubdtype(dtype, np.complexfloating):
        dset.attrs.create('_FillValue', data=np.nan + 1j * np.nan)

    if valid_min is not None:
        dset.attrs.create('valid_min', data=valid_min)

    if valid_max is not None:
        dset.attrs.create('valid_max', data=valid_max)

    # Construct the cube rasters directly from HDF5 dataset
    raster = isce3.io.Raster(f"IH5:::ID={dset.id.id}".encode("utf-8"),
                             update=True)

    return raster


def add_geolocation_grid_cubes_to_hdf5(hdf5_obj, cube_group_name, radar_grid,
                                       heights, orbit, native_doppler,
                                       grid_doppler, epsg,
                                       threshold_geo2rdr=1e-8,
                                       numiter_geo2rdr=100, delta_range=1e-8):
    if cube_group_name not in hdf5_obj:
        cube_group = hdf5_obj.create_group(cube_group_name)
    else:
        cube_group = hdf5_obj[cube_group_name]

    cube_shape = [len(heights), radar_grid.length, radar_grid.width]

    xds, yds, zds = set_create_geolocation_grid_coordinates(
        hdf5_obj, cube_group_name, radar_grid, heights, epsg)

    if epsg == 4326:
        x_coord_units = "degree_east"
        y_coord_units = "degree_north"
    else:
        x_coord_units = "meters"
        y_coord_units = "meters"

    coordinate_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'coordinateX', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Coordinate X',
        descr='X coordinates in specified EPSG code',
        units=x_coord_units)
    coordinate_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'coordinateY', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Coordinate Y',
        descr='Y coordinates in specified EPSG code',
        units=y_coord_units)
    incidence_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'incidenceAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='incidence angle',
        descr='Incidence angle is defined as the angle between the LOS '
              'vector and the normal to the ellipsoid at the target height',
        units='degrees')
    los_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector X',
        descr='East component of unit vector of LOS from target to sensor',
        units='1')
    los_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector Y',
        descr='North component of unit vector of LOS from target to sensor',
        units='1')
    along_track_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector X',
        descr='East component of unit vector along ground track',
        units='1')
    along_track_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector Y',
        descr='North component of unit vector along ground track',
        units='1')
    elevation_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'elevationAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Elevation angle',
        descr='Elevation angle is defined as the angle between the LOS vector '
              'and the normal to the ellipsoid at the sensor',
        units='degrees')
    ground_track_velocity_raster = _get_raster_from_hdf5_ds(
        cube_group, 'groundTrackVelocity', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Ground-track velocity',
        descr='Absolute value of the platform velocity scaled at the target height',
        units='meters / second')

    isce3.geometry.make_geolocation_cubes(radar_grid,
                                          heights,
                                          orbit,
                                          native_doppler,
                                          grid_doppler,
                                          epsg,
                                          coordinate_x_raster,
                                          coordinate_y_raster,
                                          incidence_angle_raster,
                                          los_unit_vector_x_raster,
                                          los_unit_vector_y_raster,
                                          along_track_unit_vector_x_raster,
                                          along_track_unit_vector_y_raster,
                                          elevation_angle_raster,
                                          ground_track_velocity_raster,
                                          threshold_geo2rdr,
                                          numiter_geo2rdr,
                                          delta_range)


def set_create_geolocation_grid_coordinates(hdf5_obj, root_ds, radar_grid,
                                            z_vect, epsg):
    rg_0 = radar_grid.starting_range
    d_rg = radar_grid.range_pixel_spacing

    rg_f = rg_0 + (radar_grid.width - 1) * d_rg
    rg_vect = np.linspace(rg_0, rg_f, radar_grid.width, dtype=np.float64)

    az_0 = radar_grid.sensing_start
    d_az = 1.0 / radar_grid.prf

    az_f = az_0 + (radar_grid.length - 1) * d_az
    az_vect = np.linspace(az_0, az_f - d_az, radar_grid.length,
                          dtype=np.float64)

    rg_coord_units = "meters"

    # seconds since ref epoch
    ref_epoch = radar_grid.ref_epoch
    ref_epoch_str = ref_epoch.isoformat().replace('T', ' ')
    az_coord_units = f'seconds since {ref_epoch_str}'

    coordinates_list = []

    # EPSG
    descr = ("EPSG code corresponding to coordinate system used" +
             " for representing geolocation grid")
    epsg_dataset_name = os.path.join(root_ds, 'epsg')
    if epsg_dataset_name in hdf5_obj:
        del hdf5_obj[epsg_dataset_name]
    epsg_dataset = hdf5_obj.create_dataset(epsg_dataset_name,
                                           data=np.array(epsg, "i4"))
    epsg_dataset.attrs["description"] = np.string_(descr)
    epsg_dataset.attrs["long_name"] = np.string_("EPSG code")

    # Slant range
    descr = "Slant range values corresponding to the geolocation grid"
    rg_dataset_name = os.path.join(root_ds, 'slantRange')
    if rg_dataset_name in hdf5_obj:
        del hdf5_obj[rg_dataset_name]
    rg_dataset = hdf5_obj.create_dataset(rg_dataset_name, data=rg_vect)
    rg_dataset.attrs["description"] = np.string_(descr)
    rg_dataset.attrs["units"] = np.string_(rg_coord_units)
    rg_dataset.attrs["long_name"] = np.string_("slant range")
    coordinates_list.append(rg_dataset)

    # Zero-doppler time
    descr = "Zero Doppler time values corresponding to the geolocation grid"
    az_dataset_name = os.path.join(root_ds, 'zeroDopplerTime')
    if az_dataset_name in hdf5_obj:
        del hdf5_obj[az_dataset_name]
    az_dataset = hdf5_obj.create_dataset(az_dataset_name, data=az_vect)
    az_dataset.attrs["description"] = np.string_(descr)
    az_dataset.attrs["units"] = np.string_(az_coord_units)
    az_dataset.attrs["long_name"] = np.string_("zero-Doppler time")
    coordinates_list.append(az_dataset)

    # Height above reference ellipsoid
    descr = "Height values above WGS84 Ellipsoid corresponding to the location grid"
    height_dataset_name = os.path.join(root_ds, 'heightAboveEllipsoid')
    if height_dataset_name in hdf5_obj:
        del hdf5_obj[height_dataset_name]
    height_dataset = hdf5_obj.create_dataset(height_dataset_name, data=z_vect)
    height_dataset.attrs['standard_name'] = np.string_("height_above_reference_ellipsoid")
    height_dataset.attrs["description"] = np.string_(descr)
    height_dataset.attrs['units'] = np.string_("meters")
    coordinates_list.append(height_dataset)

    return coordinates_list
