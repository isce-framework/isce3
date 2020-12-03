"""
collection of functions to prepare HDF5
for GSLC, GCOV, and INSAR workflows
"""

import os

import h5py
import numpy as np
from osgeo import osr

import journal
from pybind_nisar.h5 import cp_h5_meta_data
from pybind_nisar.products.readers import SLC


def run(cfg, dst: str):
    '''
    copy metadata from src hdf5 and prepare datasets
    '''
    info_channel = journal.info("h5_prep.run")
    info_channel.log('preparing HDF5')
    cp_geocode_meta(cfg, dst)
    prep_ds(cfg, dst)
    info_channel.log('successfully prepared HDF5')


def cp_geocode_meta(cfg, dst):
    '''
    Copy shared data from source HDF5 to GSLC, GCOV, INSAR
    HDF5 destinations
    Parameters:
    -----------
    cfg : dict
        Run configuration
    dst : str
        Name of destination node where data is to be copied
    '''
    is_insar = dst in ['GUNW', 'RUNW', 'RIFG']

    # unpack
    output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']
    input_hdf5 = cfg['InputFileGroup']['InputFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    if is_insar:
        secondary_hdf5 = cfg['InputFileGroup']['SecondaryFilePath']

    # rm anything and start from scratch
    try:
        os.remove(output_hdf5)
    except FileNotFoundError:
        pass

    slc = SLC(hdf5file=input_hdf5)

    # prelim setup
    common_parent_path = 'science/LSAR'
    src_h5 = h5py.File(input_hdf5, 'r', libver='latest', swmr=True)
    src_meta_path = slc.MetadataPath
    dst_meta_path = f'{common_parent_path}/{dst}/metadata'

    with h5py.File(input_hdf5, 'r', libver='latest', swmr=True) as src_h5, \
        h5py.File(output_hdf5, 'w', libver='latest', swmr=True) as dst_h5: 

        # simple copies of identification, metadata/orbit, metadata/attitude groups
        cp_h5_meta_data(src_h5, dst_h5, f'{common_parent_path}/identification',
                        excludes='productType')

        # Flag isGeocoded
        ident = dst_h5[f'{common_parent_path}/identification']
        dset = ident.create_dataset('isGeocoded', data=np.string_("True"))
        desc = "Flag to indicate radar geometry or geocoded product"
        dset.attrs["description"] = np.string_(desc)

        ident['productType'] = dst

        # copy orbit information group
        cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/orbit',
            f'{dst_meta_path}/orbit')

        # copy attitude information group
        cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/attitude',
            f'{dst_meta_path}/attitude')

        # copy radar grid information group
        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_meta_path}/geolocationGrid',
                        f'{dst_meta_path}/radarGrid',
                        renames={'coordinateX':'xCoordinates',
                            'coordinateY':'yCoordinates',
                            'zeroDopplerTime':'zeroDopplerAzimuthTime'})

        # Copy common metadata
        # TODO check differences in processingInformation
        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_meta_path}/processingInformation/algorithms',
                        f'{dst_meta_path}/processingInformation/algorithms')
        cp_h5_meta_data(src_h5, dst_h5,
                        f'{src_meta_path}/processingInformation/inputs',
                        f'{dst_meta_path}/processingInformation/inputs',
                        excludes=['l0bGranules', 'demFiles', 'zeroDopplerTime', 'slantRange'])

        inputs = [input_hdf5]
        if is_insar:
            inputs += secondary_hdf5
        input_grp = dst_h5[os.path.join(dst_meta_path, 'processingInformation/inputs')]
        dset = input_grp.create_dataset("l1SlcGranules", data=np.string_(inputs))
        desc = "List of input L1 products used"
        dset.attrs["description"] = np.string_(desc)

        # copy calibration information group and zeroDopplerTimeSpacing
        excludes = []
        if is_insar:
            excludes = ['nes0', 'elevationAntennaPattern']
        for freq in freq_pols.keys():
            frequency = f'frequency{freq}'

            # copy zeroDopplerTimeSpacing
            zero_doppler_time_spacing = src_h5[
                f'{common_parent_path}/SLC/swaths/zeroDopplerTimeSpacing'][...]
            dst_h5[f'{common_parent_path}/{dst}/grids/{frequency}'
                   '/zeroDopplerTimeSpacing/'] = zero_doppler_time_spacing

            pol_list = freq_pols[freq]
            if pol_list is None:
                continue
            for polarization in pol_list:
                cp_h5_meta_data(src_h5, dst_h5,
                                os.path.join(src_meta_path,
                                             f'calibrationInformation/{frequency}/{polarization}'),
                                os.path.join(dst_meta_path,
                                             f'calibrationInformation/{frequency}/{polarization}'),
                                excludes=excludes)
                if is_insar:
                    path = os.path.join(dst_meta_path, f'calibrationInformation/{frequency}/{polarization}')
                    descr = "Constant wrapped reference phase used to balance the interferogram"
                    _create_datasets(dst_h5[path], [0], np.float32, 'referencePhase',
                                     descr=descr, units="radians")

        # copy processingInformation
        excludes = ['nes0', 'elevationAntennaPattern']
        if is_insar:
            excludes=['frequencyA', 'frequencyB', 'azimuthChirpWeighting',
                    'effectiveVelocity', 'rangeChirpWeighting', 'slantRange', 'zeroDopplerTime']
        cp_h5_meta_data(src_h5, dst_h5,
                        os.path.join(src_meta_path, 'processingInformation/parameters'),
                        os.path.join(dst_meta_path, 'processingInformation/parameters'),
                        excludes=excludes)

        # copy product specifics
        if is_insar:
            copy_insar_meta(cfg, src_meta_path, dst_meta_path, src_h5, dst_h5)
        else:
            copy_gslc_gcov_meta(slc.SwathPath, dst, src_h5, dst_h5)


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
                        os.path.join(common_parent_path, f'{dst}/grids/frequency{freq}'),
                        excludes=['acquiredCenterFrequency', 'acquiredAzimuthBandwidth',
                                  'acquiredRangeBandwidth', 'nominalAcquisitionPRF', 'slantRange',
                                  'sceneCenterAlongTrackSpacing', 'sceneCenterGroundRangeSpacing',
                                  'HH', 'HV', 'VH', 'VV', 'RH', 'RV',
                                  'validSamplesSubSwath1', 'validSamplesSubSwath2',
                                  'validSamplesSubSwath3', 'validSamplesSubSwath4',
                                  'listOfPolarizations'],
                        renames={'processedCenterFrequency': 'centerFrequency',
                                 'processedAzimuthBandwidth': 'azimuthBandwidth',
                                 'processedRangeBandwidth': 'rangeBandwidth'})


def copy_insar_meta(cfg, src_meta_path, dst_meta_path, src_h5, dst_h5):
    '''
    Copy metadata specific to INSAR workflow
    '''

    secondary_hdf5 = cfg['InputFileGroup']['SecondaryFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    src2_h5 = h5py.File(secondary_hdf5, 'r')

    dst_proc = os.path.join(dst_meta_path, 'processingInformation/parameters')
    src_proc = os.path.join(src_meta_path, 'processingInformation/parameters')

    # Create groups in processing Information

    dst_h5.create_group(os.path.join(dst_proc, 'common'))
    dst_h5.create_group(os.path.join(dst_proc, 'reference'))
    dst_h5.create_group(os.path.join(dst_proc, 'secondary'))

    # Copy data for reference and secondary
    cp_h5_meta_data(src_h5, dst_h5, os.path.join(src_proc, 'effectiveVelocity'),
                    os.path.join(dst_proc, 'reference/effectiveVelocity'))
    cp_h5_meta_data(src2_h5, dst_h5, os.path.join(src_proc, 'effectiveVelocity'),
                    os.path.join(dst_proc, 'secondary/effectiveVelocity'))

    for freq in freq_pols.keys():
        frequency = f'frequency{freq}'
        cp_h5_meta_data(src_h5, dst_h5, os.path.join(src_proc, f'{frequency}'),
                        os.path.join(dst_proc, f'reference/{frequency}'))
        cp_h5_meta_data(src2_h5, dst_h5, os.path.join(src_proc, f'{frequency}'),
                        os.path.join(dst_proc, f'secondary/{frequency}'))

    # Copy secondary image slantRange and azimuth time (modify attributes)?
    cp_h5_meta_data(src2_h5, dst_h5, os.path.join(src_meta_path, 'geolocationGrid/slantRange'),
                    os.path.join(dst_meta_path, 'radarGrid/secondarySlantRange'))
    cp_h5_meta_data(src2_h5, dst_h5, os.path.join(src_meta_path, 'geolocationGrid/zeroDopplerTime'),
                    os.path.join(dst_meta_path, 'radarGrid/secondaryZeroDopplerAzimuthTime'))
    
    # Update these attribute with a description
    descr = "Slant range of corresponding pixels in secondary image"
    dst_h5[os.path.join(dst_meta_path, 'radarGrid/secondarySlantRange')].attrs["description"]=descr
    descr = "Zero Doppler azimuth time of corresponding pixel in secondary image"
    dst_h5[os.path.join(dst_meta_path, 'radarGrid/secondaryZeroDopplerAzimuthTime')].attrs["description"]=descr


def prep_ds(cfg, dst):
    '''
    Prepare datasets for GSLC, GCOV, INSAR workflows
    '''

    # unpack
    output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']
    dst_h5 = h5py.File(output_hdf5, 'a', libver='latest', swmr=True)

    # Fork the dataset preparation for GSLC/GCOV and GUNW
    if dst in ['GSLC', 'GCOV']:
        prep_ds_gslc_gcov(cfg, dst, dst_h5)
    else:
        prep_ds_gunw(cfg, dst, dst_h5)

    dst_h5.close()


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
        dst_parent_path = os.path.join(common_parent_path, f'{dst}/grids/frequency{freq}')

        set_get_geo_info(dst_h5, dst_parent_path, geogrids[freq])

        # GSLC specfics datasets
        if dst == 'GSLC':
            for polarization in pol_list:
                dst_grp = dst_h5[dst_parent_path]
                descr = 'Geocoded RSLC for {polarization} channel'
                _create_datasets(dst_grp, shape, ctype, polarization,
                                 descr=descr, units=None, grids="projections")

        # set GCOV polarization values (diagonal values only)
        if dst == 'GCOV':
            pol_list = [(p + p).upper() for p in pol_list]

        _add_polarization_list(dst_h5, dst, common_parent_path, freq, pol_list)


def prep_ds_gunw(cfg, dst, dst_h5):
    '''
    prepare GUNW specific datasets
    '''
    # unpack info
    common_parent_path = 'science/LSAR'

    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # Data type
    ctype = h5py.h5t.py_create(np.float64)
    ctype.commit(dst_h5['/'].id, np.string_('float64'))

    # Create datasets in the ouput hdf5
    geogrids = cfg['processing']['geocode']['geogrids']

    for freq in freq_pols.keys():
        pol_list = freq_pols[freq]
        shape = (geogrids[freq].length, geogrids[freq].width)

        dst_parent_path = os.path.join(common_parent_path, f'{dst}/grids/frequency{freq}')

        # Add x/yCoordinates, x/yCoordinatesSpacing
        set_get_geo_info(dst_h5, dst_parent_path, geogrids[freq])

        # Add list of polarizations
        _add_polarization_list(dst_h5, dst, common_parent_path, freq, pol_list)

        # Add center frequency and number of subswaths
        descr = "Center frequency of the processed image"
        _create_datasets(dst_h5[dst_parent_path], [0], np.float32, "centerFrequency",
                         descr=descr, units="Hz")
        descr = "Number of swaths of continuous imagery, due to gaps"
        _create_datasets(dst_h5[dst_parent_path], [0], np.float32, "numberOfSubSwaths",
                         descr=descr, units="Hz")

        dst_path_intf = os.path.join(dst_parent_path, 'interferogram')
        dst_path_offs = os.path.join(dst_parent_path, 'pixelOffsets')
        dst_h5.create_group(dst_path_intf)
        dst_h5.create_group(dst_path_offs)

        # Create Interferogram datasets (common to HH/VV polarizations)
        descr = "Ionosphere phase screen "
        _create_datasets(dst_h5[dst_path_intf], shape, ctype, 'ionospherePhaseScreen', chunks=(128, 128),
                         descr=descr, units="radians")
        descr = "Uncertainty of split spectrum ionosphere phase screen"
        _create_datasets(dst_h5[dst_path_intf], shape, ctype, 'ionospherePhaseScreenUncertainty', chunks=(128, 128),
                         descr=descr, units="radians")
        descr = f"Layover/shadow mask for frequency {freq} layers"
        _create_datasets(dst_h5[dst_path_intf], shape, np.uint8, 'layoverShadowMask', chunks=(128, 128),
                         descr=descr, units="DN")
        descr = " Water mask for frequency {freq} layers"
        _create_datasets(dst_h5[dst_path_intf], shape, np.uint8, 'waterMask', chunks=(128, 128),
                         descr=descr, units="DN")

        # Adding scalar datasets to interferogram
        descr = "Processed azimuth bandwidth in Hz"
        _create_datasets(dst_h5[dst_path_intf], [0], np.float32, 'azimuthBandwidth',
                         descr=descr, units="Hz")
        _create_datasets(dst_h5[dst_path_intf], [0], np.float32, 'rangeBandwidth',
                         descr=descr.replace("azimuth", "range"), units="Hz")
        descr = " Averaging window size in pixels in azimuth direction for covariance \
                  matrix estimation"
        _create_datasets(dst_h5[dst_path_intf], [0], np.uint8, 'numberOfAzimuthLooks',
                         descr=descr, units="seconds")
        _create_datasets(dst_h5[dst_path_intf], [0], np.uint8, 'numberOfRangeLooks',
                         descr=descr.replace("azimuth", "slant range"), units="seconds")
        descr = "Slant range spacing of grid. Same as difference between \
                 consecutive samples in slantRange array"
        _create_datasets(dst_h5[dst_path_intf], [0], np.uint8, 'slantRangeSpacing',
                         descr=descr, units="meters")

        # Adding scalar datasets to pixelOffsets
        descr = "Along track window size for cross-correlation"
        _create_datasets(dst_h5[dst_path_offs], [0], np.uint8, 'alongTrackWindowSize',
                         descr=descr, units="pixels")
        _create_datasets(dst_h5[dst_path_offs], [0], np.uint8, 'slantRangeWindowSize',
                         descr=descr.replace("Along track", "Slant range"), units="pixels")
        descr = "Along track skip window size for cross-correlation"
        _create_datasets(dst_h5[dst_path_offs], [0], np.uint8, 'alongTrackSkipWindowSize',
                         descr=descr, units="pixels")
        _create_datasets(dst_h5[dst_path_offs], [0], np.uint8, 'slantRangeSkipWindowSize',
                         descr=descr.replace("Along track ", "Slant range"), units="pixels")
        descr = "Along track search window size for cross-correlation"
        _create_datasets(dst_h5[dst_path_offs], [0], np.uint8, 'alongTrackSearchWindowSize',
                         descr=descr, units="pixels")
        _create_datasets(dst_h5[dst_path_offs], [0], np.uint8, 'slantRangeSearchWindowSize',
                         descr=descr.replace("Along track ", "Slant range"), units="pixels")
        descr = "Oversampling factor of the cross-correlation surface"
        _create_datasets(dst_h5[dst_path_offs], [0], np.uint8, 'correlationSurfaceOversampling',
                         descr=descr, units="unitless")
        descr = "Method used for generating pixel offsets"
        _create_datasets(dst_h5[dst_path_offs], [9], np.string_, 'crossCorrelationMethod',
        descr=descr, units=None)

        for pol in pol_list:
            intf_path = os.path.join(dst_path_intf, f'{pol}')
            offs_path = os.path.join(dst_path_offs, f'{pol}')

            dst_h5.create_group(intf_path)
            dst_h5.create_group(offs_path)

            descr = f"Unwrapped interferogram between {pol} layers"
            _create_datasets(dst_h5[intf_path], shape, ctype, 'unwrappedPhase',
                             descr=descr, units="radians")
            descr = f"Phase sigma coherence between {pol} layers"
            _create_datasets(dst_h5[intf_path], shape, ctype, 'phaseSigmaCoherence',
                             descr=descr, units="unitless")

            # Note: connected components and coherence mask are integer
            descr = f"Connected components for {pol} layer"
            _create_datasets(dst_h5[intf_path], shape, np.uint8, 'connectedComponents',
                             descr=descr, units="DN")
            descr = f"Coherence mask for {pol} layer"
            _create_datasets(dst_h5[intf_path], shape, np.uint8, 'coherenceMask',
                             descr=descr, units="DN")
            descr = f"Along track offset for {pol} layer"
            _create_datasets(dst_h5[offs_path], shape, ctype, 'alongTrackOffset',
                             descr=descr, units="meters")
            _create_datasets(dst_h5[offs_path], shape, ctype, 'slantRangeOffset',
                             descr=descr.replace("Along track", "Slant range"), units="meters")
            descr = " Correlation metric"
            _create_datasets(dst_h5[offs_path], shape, ctype, 'correlation',
                             descr=descr, units="unitless")

    # add datasets in metadata	
    dst_cal = os.path.join(common_parent_path, f'{dst}/metadata/calibrationInformation')
    dst_proc = os.path.join(common_parent_path, f'{dst}/metadata/processingInformation/parameters')
    dst_grid = os.path.join(common_parent_path, f'{dst}/metadata/radarGrid')

    descr = "Perpendicular component of the InSAR baseline"
    _create_datasets(dst_h5[dst_grid], shape, ctype, "perpendicularBaseline",
                     descr=descr, units="meters")
    _create_datasets(dst_h5[dst_grid], shape, ctype, "parallelBaseline",
                     descr=descr.replace('Perpendicular', 'Parallel'), units="meters")

    for freq in freq_pols.keys():
        dst_cal_group = f'{dst_cal}/frequency{freq}'

        descr = "Bulk along track time offset used to align reference and secondary image"
        _create_datasets(dst_h5[dst_cal_group], [0], np.float32, 'bulkAlongTrackTimeOffset',
                         descr=np.string_(descr), units="seconds")
        _create_datasets(dst_h5[dst_cal_group], [0], np.float32, 'bulkSlantRangeOffset',
                         descr=np.string_(descr.replace('along track', 'slant range')), units="seconds")

        dst_common_group = f'{dst_proc}/common/frequency{freq}'
        dst_h5.create_group(dst_common_group)

        descr = " Common Doppler bandwidth used for processing the interferogram"
        _create_datasets(dst_h5[dst_common_group], [0], np.float32, 'DopplerBandwidth',
                         descr=descr, units="Hz")
        descr = f" 2D LUT of Doppler Centroid for frequency {freq}"
        _create_datasets(dst_h5[dst_common_group], shape, ctype, "dopplerCentroid",
                         descr=descr, units="Hz")
        descr = "Number of looks applied in along track direction"
        _create_datasets(dst_h5[dst_common_group], [0], np.float32, 'numberOfAzimuthLooks',
                         descr=descr, units="unitless")
        _create_datasets(dst_h5[dst_common_group], [0], np.float32, 'numberOfRangeLooks',
                         descr=descr.replace("along track", "slant range"), units="unitless")


def _create_datasets(dst_grp, shape, ctype, dataset_name,
                     chunks=(128, 128), descr=None, units=None, grids=None):
    if len(shape) == 1:
        if ctype == np.string_:
            ds = dst_grp.create_dataset(dataset_name, data=np.string_("         "))
        else:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, data=0)
    else:
        if chunks[0] < shape[0] and chunks[1] < shape[1]:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, shape=shape, chunks=chunks)
        else:
            ds = dst_grp.create_dataset(dataset_name, dtype=ctype, shape=shape)

    ds.attrs['description'] = np.string_(descr)

    if units is not None:
        ds.attrs['units'] = np.string_(units)

    if grids is not None:
        ds.attrs['grid_mapping'] = np.string_(grids)


def _add_polarization_list(dst_h5, dst, common_parent_path, frequency, pols):
    '''
    '''
    dataset_path = os.path.join(common_parent_path, f'{dst}/grids/frequency{frequency}')
    grp = dst_h5[dataset_path]
    name = "listOfPolarizations"
    pols_array = np.array(pols, dtype="S2")
    dset = grp.create_dataset(name, data=pols_array)
    desc = f"List of polarization layers with frequency{frequency}"
    dset.attrs["description"] = np.string_(desc)

def set_get_geo_info(hdf5_obj, root_ds, geo_grid):

    epsg_code = geo_grid.epsg

    dx = geo_grid.spacing_x
    x0 = geo_grid.start_x + 0.5 * dx
    xf = x0 + (geo_grid.width - 1) * dx
    x_vect = np.linspace(x0, xf, geo_grid.width, dtype=np.float64)

    dy = geo_grid.spacing_y
    y0 = geo_grid.start_y + 0.5 * dy
    yf = y0 + (geo_grid.length - 1) * dy
    y_vect = np.linspace(y0, yf, geo_grid.length, dtype=np.float64)

    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")

    if epsg_code == 4326:
        x_coord_units = "degrees_east"
        y_coord_units = "degrees_north"
        x_standard_name = "longitude"
        y_standared_name = "latitude"
    else:
        x_coord_units = "meters"
        y_coord_units = "meters"
        x_standard_name = "projection_x_coordinate"
        y_standared_name = "projection_y_coordinate"

    # xCoordinateSpacing
    descr = (f'Nominal spacing in {x_coord_units}'
             ' between consecutive pixels')

    xds_spacing_name = os.path.join(root_ds, 'xCoordinateSpacing')
    if xds_spacing_name in hdf5_obj:
        del hdf5_obj[xds_spacing_name]
    xds_spacing = hdf5_obj.create_dataset(xds_spacing_name, data=dx)
    xds_spacing.attrs["description"] = np.string_(descr)
    xds_spacing.attrs["units"] = np.string_(x_coord_units)

    # yCoordinateSpacing
    descr = (f'Nominal spacing in {y_coord_units}'
             ' between consecutive lines')

    yds_spacing_name = os.path.join(root_ds, 'yCoordinateSpacing')
    if yds_spacing_name in hdf5_obj:
        del hdf5_obj[yds_spacing_name]
    yds_spacing = hdf5_obj.create_dataset(yds_spacing_name, data=dy)
    yds_spacing.attrs["description"] = np.string_(descr)
    yds_spacing.attrs["units"] = np.string_(y_coord_units)

    # xCoordinates
    descr = "CF compliant dimension associated with the X coordinate"
    xds_name = os.path.join(root_ds, 'xCoordinates')
    if xds_name in hdf5_obj:
        del hdf5_obj[xds_name]
    xds = hdf5_obj.create_dataset(xds_name, data=x_vect)
    xds.attrs['standard_name'] = np.string_(x_standard_name)
    xds.attrs["description"] = np.string_(descr)
    xds.attrs["units"] = np.string_(x_coord_units)

    # yCoordinates
    descr = "CF compliant dimension associated with the Y coordinate"
    yds_name = os.path.join(root_ds, 'yCoordinates')
    if yds_name in hdf5_obj:
        del hdf5_obj[yds_name]
    yds = hdf5_obj.create_dataset(yds_name, data=y_vect)
    yds.attrs['standard_name'] = np.string_(y_standared_name)
    yds.attrs["description"] = np.string_(descr)
    yds.attrs["units"] = np.string_(y_coord_units)

    coordinates_list = [xds, yds]

    try:
        for _ds in coordinates_list:
            _ds.make_scale()
    except AttributeError:
        pass

    # Associate grid mapping with data - projection created later
    projection_ds_name = os.path.join(root_ds, "projection")

    ###Create a new single int dataset for projections
    if projection_ds_name in hdf5_obj:
        del hdf5_obj[projection_ds_name]
    projds = hdf5_obj.create_dataset(projection_ds_name, (), dtype='i')
    projds[()] = epsg_code

    # Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)

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

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg_code)

    projds.attrs['grid_mapping_name'] = sr.GetName()

    ###Set up units
    ###Geodetic latitude / longitude
    if epsg_code == 4326:
        # Set up grid mapping
        projds.attrs['longitude_of_prime_meridian'] = 0.0

        projds.attrs['latitude_of_projection_origin'] = sr.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN)
        projds.attrs['longitude_of_projection_origin'] = sr.GetProjParm(osr.SRS_PP_LONGITUDE_OF_ORIGIN)

    else:
        ### UTM zones
        if ((epsg_code > 32600 and
                epsg_code < 32661) or
                (epsg_code > 32700 and
                    epsg_code < 32761)):
            # Set up grid mapping
            projds.attrs['utm_zone_number'] = epsg_code % 100

        ### Polar Stereo North
        elif epsg_code == 3413:
            # Set up grid mapping
            projds.attrs['standard_parallel'] = 70.0
            projds.attrs['straight_vertical_longitude_from_pole'] = -45.0

        ### Polar Stereo south
        elif epsg_code == 3031:
            # Set up grid mapping
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0

        ### EASE 2 for soil moisture L3
        elif epsg_code == 6933:
            # Set up grid mapping
            projds.attrs['longitude_of_central_meridian'] = 0.0
            projds.attrs['standard_parallel'] = 30.0

        ### Europe Equal Area for Deformation map (to be implemented in isce3)
        elif epsg_code == 3035:
            # Set up grid mapping
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0

        else:
            raise NotImplementedError(f'EPSG {epsg_code} waiting for implementation / not supported in ISCE3')

        # Setup common parameters
        xds.attrs['long_name'] = np.string_("x coordinate of projection")
        yds.attrs['long_name'] = np.string_("y coordinate of projection")

        projds.attrs['false_easting'] = sr.GetProjParm(osr.SRS_PP_FALSE_EASTING)
        projds.attrs['false_northing'] = sr.GetProjParm(osr.SRS_PP_FALSE_NORTHING)

        projds.attrs['latitude_of_projection_origin'] = sr.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN)
        projds.attrs['longitude_of_projection_origin'] = sr.GetProjParm(osr.SRS_PP_LONGITUDE_OF_ORIGIN)

    return yds, xds


