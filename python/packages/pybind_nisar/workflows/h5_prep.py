"""
Suite of functions to prepare
HDF5 for GSLC, GCOV, GUNW, RIFG, and RUNW
"""

import os

import h5py
import journal
import numpy as np

from osgeo import osr

import pybind_isce3 as isce3
from pybind_nisar.h5 import cp_h5_meta_data
from pybind_nisar.products.readers import SLC


def get_products_and_paths(cfg: dict) -> (dict, dict):
    '''
    Get sub-product type dict and sub-product HDF5 paths
    '''
    output_path = cfg['ProductPathGroup']['SASOutputFile']
    scratch = cfg['ProductPathGroup']['ScratchPath']
    product_type = cfg['PrimaryExecutable']['ProductType']

    # dict keying product type with list with possible product type(s)
    insar_products = ['RIFG', 'RUNW', 'GUNW']
    product_dict = {'RIFG_RUNW_GUNW': insar_products,
                    'GUNW': insar_products,
                    'RUNW': insar_products[:-1],
                    'RIFG': [insar_products[0]],
                    'GCOV': ['GCOV'],
                    'GSLC': ['GSLC'],
                    'RUNW_STANDALONE': ['RUNW'],
                    'GUNW_STANDALONE': ['GUNW']}

    # dict keying product type to dict of product type key(s) to output(s)
    # following lambda creates subproduct specific output path
    insar_path = lambda out_path, product: \
        os.path.join(os.path.dirname(out_path),
                     product + '_' + os.path.basename(out_path))
    h5_paths = {'RIFG_RUNW_GUNW': dict(zip(insar_products,
                                  [insar_path(output_path, product) for product
                                   in insar_products])),
                'GUNW': {'RIFG': f'{scratch}/RIFG.h5',
                         'RUNW': f'{scratch}/RUNW.h5', 'GUNW': output_path},
                'RUNW': {'RIFG': f'{scratch}/RIFG.h5', 'RUNW': output_path},
                'RIFG': {'RIFG': output_path},
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
    is_insar = dst in ['GUNW', 'RUNW', 'RIFG']

    # unpack info
    input_hdf5 = cfg['InputFileGroup']['InputFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    if is_insar:
        secondary_hdf5 = cfg['InputFileGroup']['SecondaryFilePath']

    if dst == "GCOV":
        dem_interp_method = cfg['processing']['dem_interpolation_method']
        geocode_algorithm = cfg['processing']['geocode']['algorithm_type']
    else:
        dem_interp_method = None
        geocode_algorithm = None

    # Remove existing HDF5 and start from scratch
    try:
        os.remove(output_hdf5)
    except FileNotFoundError:
        pass

    # Open reference slc
    ref_slc = SLC(hdf5file=input_hdf5)

    # prelim setup
    common_parent_path = 'science/LSAR'
    src_meta_path = ref_slc.MetadataPath
    dst_meta_path = f'{common_parent_path}/{dst}/metadata'

    with h5py.File(input_hdf5, 'r', libver='latest', swmr=True) as src_h5, \
            h5py.File(output_hdf5, 'w', libver='latest', swmr=True) as dst_h5:

        dst_h5.attrs['Conventions'] = np.string_("CF-1.8")

        # Copy of identification
        identification_excludes = 'productType'
        if is_insar:
            identification_excludes = ['productType', 'listOfFrequencies',
                                       'zeroDopplerStartTime', 'zeroDopplerEndTime']
        cp_h5_meta_data(src_h5, dst_h5, f'{common_parent_path}/identification',
                        excludes=identification_excludes)
        # If insar, create reference/secondary zeroDopplerStartEndTime
        if is_insar:
            # Open secondary hdf5 to copy information
            with h5py.File(secondary_hdf5, 'r', libver='latest', swmr=True) as sec_src_h5:
                src_dataset = ['zeroDopplerStartTime', 'zeroDopplerEndTime']
                dst_dataset = ['referenceZeroDopplerStartTime', 'referenceZeroDopplerEndTime']
                for src_data, dst_data in zip(src_dataset, dst_dataset):
                    cp_h5_meta_data(src_h5, dst_h5, f'{common_parent_path}/identification/{src_data}',
                                    f'{common_parent_path}/identification/{dst_data}')
                    dst_data = dst_data.replace('reference', 'secondary')
                    cp_h5_meta_data(sec_src_h5, dst_h5, f'{common_parent_path}/identification/{src_data}',
                                    f'{common_parent_path}/identification/{dst_data}')

        # Flag isGeocoded
        ident = dst_h5[f'{common_parent_path}/identification']
        is_geocoded = dst in ['GCOV', 'GSLC', 'GUNW']
        if 'isGeocoded' in ident:
            del ident['isGeocoded']
        dset = ident.create_dataset('isGeocoded',
                                    data=np.string_(str(is_geocoded)))
        desc = "Flag to indicate radar geometry or geocoded product"
        dset.attrs["description"] = np.string_(desc)

        # Assign productType
        ident['productType'] = np.string_(dst)

        # copy orbit information group
        cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/orbit',
                        f'{dst_meta_path}/orbit')

        # copy attitude information group
        cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/attitude',
                        f'{dst_meta_path}/attitude')
        if dst in ['RIFG', 'RUNW']:
            # RUNW and RIFG have no attitude group and have geolocation grid
            yds = dst_h5.create_dataset(f'{dst_meta_path}/geolocationGrid/zeroDopplerTime',
                    data = src_h5[f'{src_meta_path}/geolocationGrid/zeroDopplerTime'])
            xds = dst_h5.create_dataset(f'{dst_meta_path}/geolocationGrid/slantRange',
                    data = src_h5[f'{src_meta_path}/geolocationGrid/slantRange'])
            cp_h5_meta_data(src_h5, dst_h5,
                            f'{src_meta_path}/geolocationGrid',
                            f'{dst_meta_path}/geolocationGrid',
                            excludes=['zeroDopplerTime', 'slantRange'],
                            attach_scales_list = [yds, xds])

        # copy processingInformation/algorithms group (common across products)
        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_meta_path}/processingInformation/algorithms',
                        f'{dst_meta_path}/processingInformation/algorithms')

        if dst == "GCOV":
            algorithms_ds = (dst_meta_path +
                             'processingInformation/algorithms/geocoding')
            dst_h5.require_dataset(algorithms_ds, (), "S27",
                data=np.string_(geocode_algorithm))

            algorithms_ds = (dst_meta_path +
                             'processingInformation/algorithms/demInterpolation')
            dst_h5.require_dataset(algorithms_ds, (), "S27",
                data=np.string_(dem_interp_method))

        # copy processingInformation/inputs group
        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_meta_path}/processingInformation/inputs',
                        f'{dst_meta_path}/processingInformation/inputs',
                        excludes=['l0bGranules'])

        # Create l1SlcGranules
        inputs = [input_hdf5]
        if is_insar:
            inputs.append(secondary_hdf5)
        input_grp = dst_h5[
            os.path.join(dst_meta_path, 'processingInformation/inputs')]
        dset = input_grp.create_dataset("l1SlcGranules",
                                        data=np.string_(inputs))
        desc = "List of input L1 RSLC products used"
        dset.attrs["description"] = np.string_(desc)
        dset.attrs["long_name"] = np.string_("list of L1 RSLC products")

        # Copy processingInformation/parameters
        if dst == 'GUNW':
            exclude_args = ['frequencyA', 'frequencyB', 'azimuthChirpWeighting',
                            'effectiveVelocity', 'rangeChirpWeighting',
                            'slantRange', 'zeroDopplerTime']
        elif dst in ['RUNW', 'RIFG']:
            exclude_args = ['frequencyA', 'frequencyB',
                            'azimuthChirpWeighting',
                            'effectiveVelocity', 'rangeChirpWeighting']
        else:
            exclude_args = ['nes0', 'elevationAntennaPattern']

        cp_h5_meta_data(src_h5, dst_h5,
                        os.path.join(src_meta_path,
                                     'processingInformation/parameters'),
                        os.path.join(dst_meta_path,
                                     'processingInformation/parameters'),
                        excludes=exclude_args)

        # Copy calibrationInformation group
        exclude_args = []
        if is_insar:
            exclude_args = ['nes0', 'elevationAntennaPattern']
        for freq in freq_pols.keys():
            frequency = f'frequency{freq}'
            pol_list = freq_pols[freq]
            if pol_list is None:
                continue
            for polarization in pol_list:
                cp_h5_meta_data(src_h5, dst_h5,
                                os.path.join(src_meta_path,
                                             f'calibrationInformation/{frequency}/{polarization}'),
                                os.path.join(dst_meta_path,
                                             f'calibrationInformation/{frequency}/{polarization}'),
                                excludes=exclude_args)

        # Copy product specifics
        if is_insar:
            copy_insar_meta(cfg, dst, src_h5, dst_h5, src_meta_path)
        else:
            copy_gslc_gcov_meta(ref_slc.SwathPath, dst, src_h5, dst_h5)

        src_h5.close()
        dst_h5.close()


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
    common_parent_path = 'science/LSAR'
    dst_meta_path = os.path.join(common_parent_path, f'{dst}/metadata')

    secondary_hdf5 = cfg['InputFileGroup']['SecondaryFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # Open secondary SLC
    with h5py.File(secondary_hdf5, 'r', libver='latest',
                   swmr=True) as secondary_h5:

        dst_proc = os.path.join(dst_meta_path,
                                'processingInformation/parameters')
        src_proc = os.path.join(src_meta_path,
                                'processingInformation/parameters')

        # Create groups in processing Information
        dst_h5.create_group(os.path.join(dst_proc, 'common'))
        dst_h5.create_group(os.path.join(dst_proc, 'reference'))
        dst_h5.create_group(os.path.join(dst_proc, 'secondary'))

        # Copy data for reference and secondary
        cp_h5_meta_data(src_h5, dst_h5,
                        os.path.join(src_proc, 'effectiveVelocity'),
                        os.path.join(dst_proc, 'reference/effectiveVelocity'))
        cp_h5_meta_data(secondary_h5, dst_h5,
                        os.path.join(src_proc, 'effectiveVelocity'),
                        os.path.join(dst_proc, 'secondary/effectiveVelocity'))
        for freq in freq_pols.keys():
            frequency = f'frequency{freq}'
            cp_h5_meta_data(src_h5, dst_h5,
                            os.path.join(src_proc, f'{frequency}'),
                            os.path.join(dst_proc, f'reference/{frequency}'))
            cp_h5_meta_data(secondary_h5, dst_h5,
                            os.path.join(src_proc, f'{frequency}'),
                            os.path.join(dst_proc, f'secondary/{frequency}'))

        # Copy secondary image slantRange and azimuth time (modify attributes)
        dst_grid_path = os.path.join(dst_meta_path, 'radarGrid')
        if dst in ['RUNW', 'RIFG']:
            dst_grid_path = os.path.join(dst_meta_path, 'geolocationGrid')

        cp_h5_meta_data(secondary_h5, dst_h5, os.path.join(src_meta_path,
                                                           'geolocationGrid/slantRange'),
                        os.path.join(dst_grid_path, 'secondarySlantRange'))
        cp_h5_meta_data(secondary_h5, dst_h5, os.path.join(src_meta_path,
                                                           'geolocationGrid/zeroDopplerTime'),
                        os.path.join(dst_grid_path,
                                     'secondaryZeroDopplerAzimuthTime'))

        # Update these attribute with a description
        descr = "Slant range of corresponding pixels in secondary image"
        dst_h5[os.path.join(dst_grid_path, 'secondarySlantRange')].attrs[
            "description"] = descr
        descr = "Zero Doppler azimuth time of corresponding pixel in secondary image"
        dst_h5[os.path.join(dst_grid_path,
                            'secondaryZeroDopplerAzimuthTime')].attrs[
            "description"] = descr


def prep_ds(cfg, output_hdf5, dst):
    '''
    Prepare datasets for GSLC, GCOV,
    INSAR (GUNW, RIFG, RUNW) workflows
    '''

    # unpack
    with h5py.File(output_hdf5, 'a', libver='latest', swmr=True) as dst_h5:
        # Fork the dataset preparation for GSLC/GCOV and GUNW
        if dst in ['GSLC', 'GCOV']:
            prep_ds_gslc_gcov(cfg, dst, dst_h5)
        else:
            prep_ds_insar(cfg, dst, dst_h5)


def prep_ds_gslc_gcov(cfg, dst, dst_h5):
    '''
    Prepare datasets for GSLC and GCOV
    '''
    # unpack info
    common_parent_path = 'science/LSAR'
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # Data type
    ctype = h5py.h5t.py_create(np.complex64)
    ctype.commit(dst_h5['/'].id, np.string_('complex64'))

    # Create datasets in the ouput hdf5
    geogrids = cfg['processing']['geocode']['geogrids']
    for freq in freq_pols.keys():
        pol_list = freq_pols[freq]
        shape = (geogrids[freq].length, geogrids[freq].width)
        dst_parent_path = os.path.join(common_parent_path,
                                       f'{dst}/grids/frequency{freq}')

        yds, xds = set_get_geo_info(dst_h5, dst_parent_path, geogrids[freq])

        # GSLC specfics datasets
        if dst == 'GSLC':
            for polarization in pol_list:
                dst_grp = dst_h5[dst_parent_path]
                long_name = f'geocoded single-look complex image {polarization}'
                descr = f'Geocoded SLC image ({polarization})'
                _create_datasets(dst_grp, shape, ctype, polarization,
                                 descr=descr, units='', grids="projection",
                                 long_name=long_name, yds=yds, xds=xds)

        # set GCOV polarization values (diagonal values only)
        elif dst == 'GCOV':
            pol_list = [(p + p).upper() for p in pol_list]

        _add_polarization_list(dst_h5, dst, common_parent_path, freq, pol_list)


def prep_ds_insar(cfg, dst, dst_h5):
    '''
    prepare INSAR (GUNW, RIFG, RUNW) specific datasets
    '''

    # unpack info
    common_parent_path = 'science/LSAR'
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # Extract range and azimuth looks
    rg_looks = cfg['processing']['crossmul']['range_looks']
    az_looks = cfg['processing']['crossmul']['azimuth_looks']

    # Extract dense offsets parameters
    rg_chip = cfg['processing']['dense_offsets']['window_range']
    az_chip = cfg['processing']['dense_offsets']['window_azimuth']
    rg_search = 2 * cfg['processing']['dense_offsets']['half_search_range']
    az_search = 2 * cfg['processing']['dense_offsets']['half_search_azimuth']
    rg_skip = cfg['processing']['dense_offsets']['skip_range']
    az_skip = cfg['processing']['dense_offsets']['skip_azimuth']
    corr_ovs = cfg['processing']['dense_offsets'][
        'correlation_surface_oversampling_factor']
    offset_width = cfg['processing']['dense_offsets']['offset_width']
    offset_length = cfg['processing']['dense_offsets']['offset_length']
    gross_offset_range = cfg['processing']['dense_offsets']['gross_offset_range']
    gross_offset_azimuth = cfg['processing']['dense_offsets']['gross_offset_azimuth']

    # Create datasets in the ouput hdf5
    geogrids = cfg['processing']['geocode']['geogrids']

    # Create list of frequencies
    id_group = dst_h5['science/LSAR/identification']
    descr = "List of frequency layers available in the product"
    dset = id_group.create_dataset('listOfFrequencies',
                                   data=np.string_(list(freq_pols.keys())))
    dset.attrs["description"] = descr

    # Open reference SLC
    input_h5 = cfg['InputFileGroup']['InputFilePath']
    ref_slc = SLC(hdf5file=input_h5)
    src_h5 = h5py.File(input_h5, 'r', libver='latest', swmr=True)

    for freq in freq_pols.keys():
        pol_list = freq_pols[freq]
        # Get SLC dimension for that frequency

        # Take size of first available polarization
        dset = src_h5[os.path.join(f'{ref_slc.SwathPath}/frequency{freq}/{pol_list[0]}')]
        az_lines, rg_cols = dset.shape

        if dst in ['RUNW', 'RIFG']:
            grid_swath = 'swaths'
            shape = (az_lines // az_looks, rg_cols // rg_looks)

            # Compute dimensions for pixel offsets in radar coordinates
            if cfg['processing']['dense_offsets']['margin'] is not None:
                margin = cfg['processing']['dense_offsets']['margin']
            else:
                margin = 0

            if (gross_offset_range is not None) and (gross_offset_azimuth is not None):
                margin = max(margin, np.abs(gross_offset_range), np.abs(gross_offset_azimuth))

            margin_rg = 2 * margin + rg_search + rg_chip
            margin_az = 2 * margin + az_search + az_chip

            # If not assigned by user, compute offset length/width
            # using dense offsets parameters
            if offset_length is None:
                offset_length = (az_lines - margin_az) // az_skip
            if offset_width is None:
                offset_width = (rg_cols - margin_rg) // rg_skip

            shape_offset = (offset_length, offset_width)
        else:
            grid_swath = 'grids'
            shape = (geogrids[freq].length, geogrids[freq].width)
            shape_offset = shape

        # Create grid or swath group depending on product
        dst_h5[os.path.join(common_parent_path, f'{dst}')].create_group(
            grid_swath)
        dst_h5[os.path.join(common_parent_path,
                            f'{dst}/{grid_swath}')].create_group(
            f'frequency{freq}')
        dst_parent_path = os.path.join(common_parent_path,
                                       f'{dst}/{grid_swath}/frequency{freq}')

        # Add list of polarizations
        _add_polarization_list(dst_h5, dst, common_parent_path, freq, pol_list)

        # Add centerFrequency and number of subswaths
        descr = "Center frequency of the processed image"
        _create_datasets(dst_h5[dst_parent_path], [0], np.float32,
                         "centerFrequency",
                         descr=descr, units="Hz", data=1,
                         long_name="center frequency")
        descr = "Number of swaths of continuous imagery, due to gaps"
        _create_datasets(dst_h5[dst_parent_path], [0], np.uint8,
                         "numberOfSubSwaths",
                         descr=descr, units=" ", data=1,
                         long_name="number of subswaths")

        # Create path to interferogram and pixelOffsets
        dst_path_intf = os.path.join(dst_parent_path, 'interferogram')
        dst_path_offs = os.path.join(dst_parent_path, 'pixelOffsets')
        dst_h5.create_group(dst_path_intf)
        dst_h5.create_group(dst_path_offs)

        if dst in ['RIFG', 'RUNW']:

            # Generate slantRange and Azimuth time (for RIFG and RUNW only)
            slant_range = src_h5[f'{ref_slc.SwathPath}/frequency{freq}/slantRange'][()]
            doppler_time = src_h5[f'{ref_slc.SwathPath}/zeroDopplerTime'][()]
            rg_spacing = src_h5[f'{ref_slc.SwathPath}/frequency{freq}/slantRangeSpacing'][()]
            az_spacing = src_h5[f'{ref_slc.SwathPath}/zeroDopplerTimeSpacing'][()]

            # TO DO: This is valid for odd number of looks. For R1 extend this to even number of looks
            idx_rg = np.arange(int(len(slant_range) / rg_looks) * rg_looks)[
                     ::rg_looks] + int(rg_looks / 2)
            idx_az = np.arange(int(len(doppler_time) / az_looks) * az_looks)[
                     ::az_looks] + int(az_looks / 2)

            descr = "CF compliant dimension associated with slant range"
            id_group = dst_h5[os.path.join(common_parent_path,
                                           f'{dst}/{grid_swath}/frequency{freq}/interferogram')]
            dset = id_group.create_dataset('slantRange',
                                           data=slant_range[idx_rg])
            dset.attrs["description"] = descr
            dset.attrs["units"] = np.string_("meters")
            dset.attrs["long_name"] = np.string_("slant range")

            descr = descr.replace("slant range", "azimuth time")
            dset = id_group.create_dataset('zeroDopplerTime',
                                           data=doppler_time[idx_az])
            dset.attrs["description"] = descr
            dset.attrs["units"] = src_h5[f'{ref_slc.SwathPath}/zeroDopplerTime'].attrs["units"]
            src_h5.close()

            # Allocate slant range and azimuth spacing
            descr = "Slant range spacing of grid. Same as difference between \
                                     consecutive samples in slantRange array"
            _create_datasets(dst_h5[dst_path_intf], [0], np.float64,
                             'slantRangeSpacing',
                             descr=descr, units="meters", data=rg_looks*rg_spacing,
                             long_name="slant range spacing")
            descr = "Time interval in the along track direction for raster layers. " \
                    "This is the same as the spacing between consecutive entries in " \
                    "zeroDopplerTime array"
            _create_datasets(dst_h5[dst_path_intf], [0], np.float32,
                             'zeroDopplerTimeSpacing',
                             descr=descr, units="seconds", data=az_looks*az_spacing,
                             long_name="zero doppler time spacing")

            descr = "Slant range spacing of offset grid"
            _create_datasets(dst_h5[dst_path_offs], [0], np.float64,
                             'slantRangeSpacing',
                             descr=descr, units="meters", data=rg_skip*rg_spacing,
                             long_name="slant range spacing")
            descr = "Along track spacing of the offset grid"
            _create_datasets(dst_h5[dst_path_offs], [0], np.float32,
                             'zeroDopplerTimeSpacing',
                             descr=descr, units="seconds", data=az_skip*az_spacing,
                             long_name="zero doppler time spacing")

            descr = "Nominal along track spacing in meters between consecutive lines" \
                    "near mid swath of the interferogram image"
            _create_datasets(dst_h5[dst_parent_path], [0], np.float32,
                             "sceneCenterAlongTrackSpacing",
                             descr=descr, units="meters", data=1,
                             long_name="scene center along track spacing")
            descr = descr.replace("Nominal along track",
                                  "Nominal ground range").replace('lines',
                                                                  'pixels')
            _create_datasets(dst_h5[dst_parent_path], [0], np.float32,
                             "sceneCenterGroundRangeSpacing",
                             descr=descr, units="meters", data=1,
                             long_name="scene center ground range spacing")

            # Valid subsamples: to be copied from RSLC or need to be created from scratch?
            descr = " First and last valid sample in each line of 1st subswath"
            _create_datasets(dst_h5[dst_parent_path], [0], np.uint8,
                             "validSubSamplesSubSwath1",
                             descr=descr, units=" ", data=1,
                             long_name="valid samples sub swath 1")
            _create_datasets(dst_h5[dst_parent_path], [0], np.uint8,
                             "validSubSamplesSubSwath2",
                             descr=descr.replace('1st', '2nd'), units=" ",
                             data=1,
                             long_name="valid samples sub swath 2")
            _create_datasets(dst_h5[dst_parent_path], [0], np.uint8,
                             "validSubSamplesSubSwath3",
                             descr=descr.replace('1st', '3rd'), units=" ",
                             data=1,
                             long_name="valid samples sub swath 3")
            _create_datasets(dst_h5[dst_parent_path], [0], np.uint8,
                             "validSubSamplesSubSwath4",
                             descr=descr.replace('1st', '4th'), units=" ",
                             data=1,
                             long_name="valid samples sub swath 4")

        descr = "Processed azimuth bandwidth in Hz"
        _create_datasets(dst_h5[dst_parent_path], [0], np.float32,
                         'azimuthBandwidth',
                         descr=descr, units="Hz", data=1,
                         long_name="azimuth bandwidth")
        _create_datasets(dst_h5[dst_parent_path], [0], np.float32,
                         'rangeBandwidth',
                         descr=descr.replace("azimuth", "range"), units="Hz",
                         data=1,
                         long_name="range bandwidth")

        # Adding polarization-dependent datasets to interferogram and pixelOffsets
        for pol in pol_list:
            intf_path = os.path.join(dst_path_intf, f'{pol}')
            offs_path = os.path.join(dst_path_offs, f'{pol}')

            dst_h5.create_group(intf_path)
            dst_h5.create_group(offs_path)

            grids_val = None
            if dst == "GUNW":
                dst_geo_path = f'{dst_parent_path}/interferogram/{pol}'
                set_get_geo_info(dst_h5, dst_geo_path, geogrids[freq])
                grids_val = 'projection'

            if dst in ['GUNW', 'RUNW']:
                descr = f"Connected components for {pol} layer"
                _create_datasets(dst_h5[intf_path], shape, np.uint8,
                                 'connectedComponents',
                                 descr=descr, units=" ", grids=grids_val,
                                 long_name='connected components')
                descr = f"Unwrapped interferogram between {pol} layers"
                _create_datasets(dst_h5[intf_path], shape, np.float32,
                                 'unwrappedPhase',
                                 descr=descr, units="radians", grids=grids_val,
                                 long_name='unwrapped phase')
                descr = f"Phase sigma coherence between {pol} layers"
                _create_datasets(dst_h5[intf_path], shape, np.float32,
                                 'coherenceMagnitude',
                                 descr=descr, units=" ", grids=grids_val,
                                 long_name='coherence magnitude')
                descr = "Ionosphere phase screen"
                _create_datasets(dst_h5[intf_path], shape, np.float32,
                                 'ionospherePhaseScreen',
                                 chunks=(128, 128),
                                 descr=descr, units="radians", grids=grids_val,
                                 long_name='ionosphere phase screen')
                descr = "Uncertainty of split spectrum ionosphere phase screen"
                _create_datasets(dst_h5[intf_path], shape, np.float32,
                                 'ionospherePhaseScreenUncertainty',
                                 chunks=(128, 128),
                                 descr=descr, units="radians", grids=grids_val,
                                 long_name='ionosphere phase screen uncertainty')
                if dst == "GUNW":
                    descr = f"Coherence mask for {pol} layer"
                    _create_datasets(dst_h5[intf_path], shape, np.float32,
                                     'coherenceMask',
                                     descr=descr, units=" ", grids=grids_val,
                                     long_name='coherence mask')
            else:
                descr = f"Interferogram between {pol} layers"
                _create_datasets(dst_h5[intf_path], shape, np.complex64,
                                 "wrappedInterferogram",
                                 chunks=(128, 128),
                                 descr=descr, units="radians",
                                 long_name='wrapped phase')
                if (az_looks, rg_looks) != (1, 1):
                    descr = f"Coherence between {pol} layers"
                    _create_datasets(dst_h5[intf_path], shape, np.float32,
                                     "coherenceMagnitude",
                                     chunks=(128, 128),
                                     descr=descr, units=None,
                                     long_name='coherence magnitude')

            descr = f"Along track offset for {pol} layer"
            _create_datasets(dst_h5[offs_path], shape_offset, np.float32,
                             'alongTrackOffset',
                             descr=descr, units="meters",
                             long_name='along track offset')
            _create_datasets(dst_h5[offs_path], shape_offset, np.float32,
                             'slantRangeOffset',
                             descr=descr.replace("Along track", "Slant range"),
                             units="meters",
                             long_name='slant range offset')
            descr = " Quality metric"
            _create_datasets(dst_h5[offs_path], shape_offset, np.float32,
                             'quality',
                             descr=descr, units=" ", long_name='quality')

        # Adding layover-shadow mask
        if dst in ['GUNW']:
            descr = f"Layover Shadow mask for frequency{freq} layer, 1 - Radar Shadow. 2 - Radar Layover. 3 - Both"
            _create_datasets(dst_h5[dst_path_intf], shape, np.byte,
                             'layoverShadowMask',
                             descr=descr, units=" ", grids=grids_val,
                             long_name='layover shadow mask')

        # Add datasets in metadata
        dst_cal = os.path.join(common_parent_path,
                               f'{dst}/metadata/calibrationInformation')
        dst_proc = os.path.join(common_parent_path,
                                f'{dst}/metadata/processingInformation/parameters')
        dst_grid = os.path.join(common_parent_path, f'{dst}/metadata/radarGrid')

        if dst in ['RUNW', 'RIFG']:
            dst_grid = os.path.join(common_parent_path,
                                    f'{dst}/metadata/geolocationGrid')

        # Add parallel and perpendicular component of baseline.
        # TO DO (R2): Define dimension of baseline LUTs
        descr = "Perpendicular component of the InSAR baseline"
        _create_datasets(dst_h5[dst_grid], shape, np.float64,
                         "perpendicularBaseline",
                         descr=descr, units="meters",
                         long_name='perpendicular baseline')
        _create_datasets(dst_h5[dst_grid], shape, np.float64,
                         "parallelBaseline",
                         descr=descr.replace('Perpendicular', 'Parallel'),
                         units="meters",
                         long_name='parallel baseline')

        dst_cal_group = f'{dst_cal}/frequency{freq}'

        descr = "Bulk along track time offset used to align reference and secondary image"

        _create_datasets(dst_h5[dst_cal_group], [0], np.float32,
                         "bulkAlongTrackTimeOffset",
                         descr=np.string_(descr), units="seconds", data=1,
                         long_name='bulk along track time offset')
        _create_datasets(dst_h5[dst_cal_group], [0], np.float32,
                         "bulkSlantRangeOffset",
                         descr=np.string_(
                             descr.replace('along track time', 'slant range')),
                         units="meters",
                         data=1, long_name='bulk slant range offset')

        # Add datasets in processingInformation/parameters/common
        dst_common_group = f'{dst_proc}/common/frequency{freq}'
        dst_h5.create_group(dst_common_group)

        # Create interferogram and pixel offset groups in
        # processingInformation/parameters/common
        dst_common_intf = os.path.join(dst_common_group, 'interferogram')
        dst_common_offs = os.path.join(dst_common_group, 'pixelOffsets')

        dst_h5.create_group(dst_common_intf)
        dst_h5.create_group(dst_common_offs)

        descr = " Common Doppler bandwidth used for processing the interferogram"
        _create_datasets(dst_h5[dst_common_group], [0], np.float64,
                         "dopplerBandwidth",
                         descr=descr, units="Hz", data=1,
                         long_name='doppler bandwidth')
        descr = f" 2D LUT of Doppler Centroid for frequency {freq}"
        _create_datasets(dst_h5[dst_common_group], shape, np.float64,
                         "dopplerCentroid",
                         descr=descr, units="Hz", data=1,
                         long_name='doppler centroid')
        descr = "Number of looks applied in along track direction"
        _create_datasets(dst_h5[dst_common_intf], [0], np.uint8,
                         "numberOfAzimuthLooks",
                         descr=descr, units=" ", data=int(az_looks),
                         long_name='number of azimuth looks')
        _create_datasets(dst_h5[dst_common_intf], [0], np.uint8,
                         "numberOfRangeLooks",
                         descr=descr.replace("along track", "slant range"),
                         units=" ",
                         data=int(rg_looks), long_name='number of range looks')

        # Adding scalar datasets to pixelOffsets group
        descr = "Along track window size for cross-correlation"
        _create_datasets(dst_h5[dst_common_offs], [0], np.uint8,
                         'alongTrackWindowSize',
                         descr=descr, units=" ", data=az_chip,
                         long_name='along track window size')
        _create_datasets(dst_h5[dst_common_offs], [0], np.uint8,
                         'slantRangeWindowSize',
                         descr=descr.replace("Along track", "Slant range"),
                         units=" ", data=rg_chip,
                         long_name="slant range window size")
        descr = "Along track skip window size for cross-correlation"
        _create_datasets(dst_h5[dst_common_offs], [0], np.uint8,
                         'alongTrackSkipWindowSize',
                         descr=descr, units=" ", data=az_skip,
                         long_name='along track skip window size')
        _create_datasets(dst_h5[dst_common_offs], [0], np.uint8,
                         'slantRangeSkipWindowSize',
                         descr=descr.replace("Along track ", "Slant range"),
                         units=" ", data=rg_skip,
                         long_name="slant range skip window size")
        descr = "Along track search window size for cross-correlation"
        _create_datasets(dst_h5[dst_common_offs], [0], np.uint8,
                         'alongTrackSearchWindowSize',
                         descr=descr, units=" ", data=az_search,
                         long_name="along track skip window size")
        _create_datasets(dst_h5[dst_common_offs], [0], np.uint8,
                         'slantRangeSearchWindowSize',
                         descr=descr.replace("Along track ", "Slant range"),
                         units=" ", data=rg_search,
                         long_name="slant range search window size")
        descr = "Oversampling factor of the cross-correlation surface"
        _create_datasets(dst_h5[dst_common_offs], [0], np.uint8,
                         'correlationSurfaceOversampling',
                         descr=descr, units=" ", data=corr_ovs,
                         long_name='correlation surface oversampling')
        descr = "Method used for generating pixel offsets"
        _create_datasets(dst_h5[dst_common_offs], [9], np.string_,
                         'crossCorrelationMethod',
                         descr=descr, units=None, data=1,
                         long_name='cross correlation method')

        if dst == "RIFG":
            descr = "Reference elevation above WGS84 Ellipsoid used for flattening"
            _create_datasets(dst_h5[dst_common_group], [0], np.float32,
                             "referenceFlatteningElevation",
                             descr=descr, units="meters", data=1,
                             long_name='reference flattening elevation')

        for pol in pol_list:
            cal_path = os.path.join(dst_cal_group, f"{pol}")
            descr = "Constant wrapped reference phase used to balance the interferogram"
            _create_datasets(dst_h5[cal_path], [0], np.float32,
                             "referencePhase",
                             descr=descr, units="radians", data=1,
                             long_name='reference phase')


def _create_datasets(dst_grp, shape, ctype, dataset_name,
                     chunks=(128, 128), descr=None, units=None, grids=None,
                     data=None, standard_name=None, long_name=None,
                     yds=None, xds=None):
    if len(shape) == 1:
        if ctype == np.string_:
            ds = dst_grp.create_dataset(dataset_name,
                                        data=np.string_("         "))
        else:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, data=data)
    else:
        # temporary fix for CUDA geocode insar's inability to direct write to
        # HDF5 with chunks (see https://github-fn.jpl.nasa.gov/isce-3/isce/issues/813 for details)
        create_with_chunks = (chunks[0] < shape[0] and chunks[1] < shape[1]) and \
            ('GUNW' not in dst_grp.name)
        if create_with_chunks:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, shape=shape,
                                        chunks=chunks)
        else:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, shape=shape)

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


def _add_polarization_list(dst_h5, dst, common_parent_path, frequency, pols):
    '''
    Add list of processed polarizations
    '''
    dataset_path = os.path.join(common_parent_path,
                                f'{dst}/grids/frequency{frequency}')

    if dst in ['RUNW', 'RIFG']:
        dataset_path = os.path.join(common_parent_path,
                                    f'{dst}/swaths/frequency{frequency}')

    grp = dst_h5[dataset_path]
    name = "listOfPolarizations"
    pols_array = np.array(pols, dtype="S2")
    dset = grp.create_dataset(name, data=pols_array)
    desc = f"List of polarization layers with frequency{frequency}"
    dset.attrs["description"] = np.string_(desc)


def set_get_geo_info(hdf5_obj, root_ds, geo_grid, z_vect=None, flag_cube=False):
    epsg_code = geo_grid.epsg

    dx = geo_grid.spacing_x
    x0 = geo_grid.start_x + 0.5 * dx
    xf = x0 + (geo_grid.width - 1) * dx
    x_vect = np.linspace(x0, xf, geo_grid.width, dtype=np.float64)

    dy = geo_grid.spacing_y
    y0 = geo_grid.start_y + 0.5 * dy
    yf = y0 + (geo_grid.length - 1) * dy
    y_vect = np.linspace(y0, yf - dy, geo_grid.length, dtype=np.float64)

    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")

    if epsg_code == 4326:
        x_coord_units = "degree_east"
        y_coord_units = "degree_north"
        x_standard_name = "longitude"
        y_standard_name = "latitude"
    else:
        x_coord_units = "m"
        y_coord_units = "m"
        x_standard_name = "projection_x_coordinate"
        y_standard_name = "projection_y_coordinate"

    if flag_cube:
        # EPSG
        descr = ("EPSG code corresponding to coordinate system used" +
                 " for representing radar grid")
        epsg_dataset_name = os.path.join(root_ds, 'epsg')
        if epsg_dataset_name in hdf5_obj:
            del hdf5_obj[epsg_dataset_name]
        epsg_dataset = hdf5_obj.create_dataset(epsg_dataset_name,
                                               data=np.array(epsg_code, "i4"))
        epsg_dataset.attrs["description"] = np.string_(descr)
        epsg_dataset.attrs["units"] = ""
        epsg_dataset.attrs["long_name"] = np.string_("EPSG code")
    else:
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
    if not flag_cube:
        descr = "CF compliant dimension associated with the X coordinate"
    else:
        descr = ('X coordinate values corresponding'
                 ' to the radar grid')
    xds_name = os.path.join(root_ds, 'xCoordinates')
    if xds_name in hdf5_obj:
        del hdf5_obj[xds_name]
    xds = hdf5_obj.create_dataset(xds_name, data=x_vect)
    xds.attrs['standard_name'] = x_standard_name
    xds.attrs["description"] = np.string_(descr)
    xds.attrs["units"] = np.string_(x_coord_units)
    xds.attrs["long_name"] = np.string_("x coordinate")

    # yCoordinates
    if not flag_cube:
        descr = "CF compliant dimension associated with the Y coordinate"
    else:
        descr = ('Y coordinate values corresponding'
                 ' to the radar grid')

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
        descr = "Height values above WGS84 Ellipsoid corresponding to the radar grid"
        zds_name = os.path.join(root_ds, 'heightAboveEllipsoid')
        if zds_name in hdf5_obj:
            del hdf5_obj[zds_name]
        zds = hdf5_obj.create_dataset(zds_name, data=z_vect)
        zds.attrs['standard_name'] = np.string_(
            "height_above_reference_ellipsoid")
        yds.attrs["description"] = np.string_(descr)
        zds.attrs['units'] = np.string_("m")
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
                                 numiter_geo2rdr=100, delta_range=1e-8,
                                 epsg_los_and_along_track_vectors=0):
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
        descr='',
        units='meter')
    azimuth_time_raster = _get_raster_from_hdf5_ds(
        cube_group, 'zeroDopplerAzimuthTime', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='zero-Doppler azimuth time',
        descr='Zero doppler azimuth time in seconds',
        units=az_coord_units)
    incidence_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'incidenceAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='incidence angle',
        descr='Incidence angle is defined as angle between LOS vector and normal at the target',
        units='degrees')
    los_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector X',
        descr='East component of unit vector of LOS from target to sensor',
        units='')
    los_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector Y',
        descr='North component of unit vector of LOS from target to sensor',
        units='')
    along_track_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector X',
        descr='East component of unit vector along ground track',
        units='')
    along_track_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector Y',
        descr='North component of unit vector along ground track',
        units='')
    elevation_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'elevationAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Elevation angle',
        descr='Elevation angle is defined as angle between LOS vector and norm at the sensor',
        units='degrees')

    isce3.geometry.make_radar_grid_cubes(radar_grid,
                                         geogrid,
                                         heights,
                                         orbit,
                                         native_doppler,
                                         grid_doppler,
                                         epsg_los_and_along_track_vectors,
                                         slant_range_raster,
                                         azimuth_time_raster,
                                         incidence_angle_raster,
                                         los_unit_vector_x_raster,
                                         los_unit_vector_y_raster,
                                         along_track_unit_vector_x_raster,
                                         along_track_unit_vector_y_raster,
                                         elevation_angle_raster,
                                         threshold_geo2rdr,
                                         numiter_geo2rdr,
                                         delta_range)


def _get_raster_from_hdf5_ds(group, ds_name, dtype, shape,
                             zds=None, yds=None, xds=None, standard_name=None,
                             long_name=None, descr=None,
                             units=None, fill_value=None,
                             valid_min=None, valid_max=None):
    # remove dataset if it already exists
    if ds_name in group:
        del group[ds_name]

    # create dataset
    dset = group.create_dataset(ds_name, dtype=np.float64, shape=shape)

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
                                       numiter_geo2rdr=100, delta_range=1e-8,
                                       epsg_los_and_along_track_vectors=0):
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
        x_coord_units = "meter"
        y_coord_units = "meter"

    coordinate_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'coordinateX', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Coordinate X',
        descr='X coordinate in specified EPSG code',
        units=x_coord_units)
    coordinate_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'coordinateY', np.float64, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Coordinate Y',
        descr='Y coordinate in specified EPSG code',
        units=y_coord_units)
    incidence_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'incidenceAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='incidence angle',
        descr='Incidence angle is defined as angle between LOS vector and normal at the target',
        units='degrees')
    los_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector X',
        descr='East component of unit vector of LOS from target to sensor',
        units='')
    los_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'losUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='LOS unit vector Y',
        descr='North component of unit vector of LOS from target to sensor',
        units='')
    along_track_unit_vector_x_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorX', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector X',
        descr='East component of unit vector along ground track',
        units='')
    along_track_unit_vector_y_raster = _get_raster_from_hdf5_ds(
        cube_group, 'alongTrackUnitVectorY', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Along-track unit vector Y',
        descr='North component of unit vector along ground track',
        units='')
    elevation_angle_raster = _get_raster_from_hdf5_ds(
        cube_group, 'elevationAngle', np.float32, cube_shape,
        zds=zds, yds=yds, xds=xds,
        long_name='Elevation angle',
        descr='Elevation angle is defined as angle between LOS vector and norm at the sensor',
        units='degrees')

    isce3.geometry.make_geolocation_cubes(radar_grid,
                                          heights,
                                          orbit,
                                          native_doppler,
                                          grid_doppler,
                                          epsg,
                                          epsg_los_and_along_track_vectors,
                                          coordinate_x_raster,
                                          coordinate_y_raster,
                                          incidence_angle_raster,
                                          los_unit_vector_x_raster,
                                          los_unit_vector_y_raster,
                                          along_track_unit_vector_x_raster,
                                          along_track_unit_vector_y_raster,
                                          elevation_angle_raster,
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

    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")

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
    epsg_dataset.attrs["units"] = ""
    epsg_dataset.attrs["long_name"] = np.string_("EPSG code")

    # Slant range
    descr = "Slant range dimension corresponding to calibration records"
    rg_dataset_name = os.path.join(root_ds, 'slantRange')
    if rg_dataset_name in hdf5_obj:
        del hdf5_obj[rg_dataset_name]
    rg_dataset = hdf5_obj.create_dataset(rg_dataset_name, data=rg_vect)
    rg_dataset.attrs["description"] = np.string_(descr)
    rg_dataset.attrs["units"] = np.string_(rg_coord_units)
    rg_dataset.attrs["long_name"] = np.string_("slant-range")
    coordinates_list.append(rg_dataset)

    # Zero-doppler time
    descr = "Zero doppler time dimension corresponding to calibration records"
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
    height_dataset.attrs['units'] = np.string_("m")
    coordinates_list.append(height_dataset)

    return coordinates_list
