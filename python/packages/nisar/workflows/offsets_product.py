import pathlib
import time

import h5py
import isce3
import journal
import numpy as np
from nisar.products.readers import SLC
from nisar.workflows import prepare_insar_hdf5
from nisar.workflows.compute_stats import (compute_stats_real_data,
                                           compute_stats_real_hdf5_dataset)
from nisar.workflows.dense_offsets import create_empty_dataset
from nisar.workflows.helpers import (copy_raster, get_cfg_freq_pols,
                                     get_ground_track_velocity_product)
from nisar.workflows.offsets_product_runconfig import OffsetsProductRunConfig
from nisar.products.insar.product_paths import ROFFGroupsPaths
from nisar.workflows.yaml_argparse import YamlArgparse
from osgeo import gdal


def run(cfg: dict, output_hdf5: str = None):
    '''
    Wrapper to run offsets product generation
    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters
    output_hdf5: str
        File path to HDF5 ROFF product to store offset layers
    '''

    # Pull parameters from cfg
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    offs_params = cfg['processing']['offsets_product']
    coreg_slc_path = pathlib.Path(offs_params['coregistered_slc_path'])

    # Initialize parameters shared between frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)

    # Instantiate ROFF obj to easily access ROFF datasets
    roff_obj = ROFFGroupsPaths()

    # Info and error channel
    error_channel = journal.error('offsets_product.run')
    info_channel = journal.info('offsets_product.run')
    info_channel.log('Start offsets product generation')

    # Check if use GPU
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])
    if use_gpu:
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)
    else:
        err_str = "Currently ISCE3 supports only GPU cross-correlation"
        error_channel.log(err_str)
        raise NotImplementedError(err_str)

    # Get the slant range and zero doppler time spacing
    ref_radar_grid = ref_slc.getRadarGrid()
    ref_slant_range_spacing = ref_radar_grid.range_pixel_spacing
    ref_zero_doppler_time_spacing = ref_radar_grid.az_time_interval

    t_all = time.time()

    # Generate the ground track velocity file that has the same
    # dimension with frequencyA of the reference RSCL
    output_dir = scratch_path / 'offsets_product'
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    info_channel.log('Produce the ground track velocity product')

    # Pull the slant range and zero doppler time of the ROFF product
    # at frequencyA
    with h5py.File(output_hdf5, 'r', libver='latest', swmr=True) as dst_h5:
        pixel_offsets_path = f'{roff_obj.SwathsPath}/frequencyA/pixelOffsets'
        slant_range = dst_h5[f'{pixel_offsets_path}/slantRange'][()]
        zero_doppler_time = \
            dst_h5[f'{pixel_offsets_path}/zeroDopplerTime'][()]

    ground_track_velocity_file = get_ground_track_velocity_product(ref_slc,
                                                                   slant_range,
                                                                   zero_doppler_time,
                                                                   dem_file,
                                                                   output_dir)

    info_channel.log('Finish producing the ground track velocity product')
    with h5py.File(output_hdf5, 'a', libver='latest', swmr=True) as dst_h5:
        # Loop over frequencies and polarizations
        for freq, _, pol_list in get_cfg_freq_pols(cfg):
            off_scratch = scratch_path / f'offsets_product/freq{freq}'

            for pol in pol_list:
                out_dir = off_scratch / pol
                out_dir.mkdir(parents=True, exist_ok=True)

                # Create a memory-mappable (ENVI) version of the ref SLC
                copy_raster(ref_hdf5, freq, pol,
                            offs_params['lines_per_block'],
                            str(out_dir / 'reference'), file_type='ENVI')
                ref_str = f'HDF5:{ref_hdf5}:/{ref_slc.slcPath(freq, pol)}'
                ref_raster = isce3.io.Raster(ref_str)

                # Create a memory mappable version (ENVI) of secondary
                if coreg_slc_path.is_file():
                    sec_path = str(out_dir / 'secondary')
                    copy_raster(sec_hdf5, freq, pol,
                                offs_params['lines_per_block'],
                                sec_path, file_type='ENVI')
                else:
                    sec_path = str(coreg_slc_path /
                                   f'coarse_resample_slc/freq{freq}/{pol}/coregistered_secondary.slc')
                sec_raster = isce3.io.Raster(sec_path)

                # Loop over offset layers to set Ampcor parameters
                layer_keys = [key for key in offs_params.keys() if
                              key.startswith('layer')]
                # If no layer is found, throw an exception
                if not layer_keys:
                    err_str = "No offset layer specified; at least one layer is required"
                    error_channel.log(err_str)
                    raise ValueError(err_str)

                for key in layer_keys:
                    # Create and initialize Ampcor object (only GPU for now)
                    if use_gpu:
                        ampcor = isce3.cuda.matchtemplate.PyCuAmpcor()
                        ampcor.deviceID = cfg['worker']['gpu_id']
                        ampcor.useMmap = 1

                    # Set parameters related to reference/secondary RSLC
                    ampcor.referenceImageName = str(out_dir / 'reference')
                    ampcor.referenceImageHeight = ref_raster.length
                    ampcor.referenceImageWidth = ref_raster.width
                    ampcor.secondaryImageName = sec_path
                    ampcor.secondaryImageHeight = sec_raster.length
                    ampcor.secondaryImageWidth = sec_raster.width

                    # Create a layer directory and set layer-dependent params
                    layer_scratch_path = out_dir / key
                    layer_scratch_path.mkdir(parents=True, exist_ok=True)
                    lay_cfg = offs_params[key]

                    # Set parameters depending on the layer
                    ampcor.windowSizeWidth = lay_cfg['window_range']
                    ampcor.windowSizeHeight = lay_cfg['window_azimuth']
                    ampcor.halfSearchRangeAcross = lay_cfg[
                            'half_search_range']
                    ampcor.halfSearchRangeDown = lay_cfg[
                            'half_search_azimuth']
                    ampcor = set_ampcor_params(offs_params, ampcor)

                    # Create empty datasets to store Ampcor results
                    ampcor.offsetImageName = str(
                            layer_scratch_path / 'dense_offsets')
                    ampcor.grossOffsetImageName = str(
                            layer_scratch_path / 'gross_offset')
                    ampcor.snrImageName = str(layer_scratch_path / 'snr')
                    ampcor.covImageName = str(layer_scratch_path / 'covariance')
                    ampcor.corrImageName = str(layer_scratch_path/ 'correlation_peak')

                    create_empty_dataset(str(layer_scratch_path / 'dense_offsets'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 2,
                                         gdal.GDT_Float32)
                    create_empty_dataset(str(layer_scratch_path / 'gross_offsets'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 2,
                                         gdal.GDT_Float32)
                    create_empty_dataset(str(layer_scratch_path / 'snr'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 1,
                                         gdal.GDT_Float32)
                    create_empty_dataset(str(layer_scratch_path / 'covariance'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 3,
                                         gdal.GDT_Float32)
                    create_empty_dataset(str(layer_scratch_path / 'correlation_peak'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 1,
                                         gdal.GDT_Float32)

                    # Run ampcor and delete ampcor object after is done
                    ampcor.runAmpcor()
                    del ampcor

                    pixel_offsets_path = f'{roff_obj.SwathsPath}/frequency{freq}/pixelOffsets'
                    prod_path = f'{pixel_offsets_path}/{pol}/{key}'

                    # Write datasets
                    along_track_offset_ds =  dst_h5[f'{prod_path}/alongTrackOffset']
                    write_along_track_offsets_data(str(layer_scratch_path / 'dense_offsets'),
                               along_track_offset_ds,
                               1, offs_params['lines_per_block'],
                               ground_track_velocity_file,
                               ref_zero_doppler_time_spacing)

                    slant_range_ds = dst_h5[f'{prod_path}/slantRangeOffset']
                    write_slant_range_offsets_data(
                        str(layer_scratch_path / 'dense_offsets'),
                        slant_range_ds,
                        2, offs_params['lines_per_block'],
                        ref_slant_range_spacing)

                    # Write the offsets covariance data
                    _write_offsets_covariance_data(
                        str(layer_scratch_path / 'covariance'),
                        dst_h5[f'{prod_path}/alongTrackOffsetVariance'],
                        dst_h5[f'{prod_path}/slantRangeOffsetVariance'],
                        dst_h5[f'{prod_path}/crossOffsetVariance'],
                        offs_params['lines_per_block'],
                        ground_track_velocity_file,
                        ref_zero_doppler_time_spacing,
                        ref_slant_range_spacing)

                    write_data(str(layer_scratch_path / 'snr'),
                               dst_h5[f'{prod_path}/snr'],
                               1, offs_params['lines_per_block'])
                    write_data(str(layer_scratch_path / 'correlation_peak'),
                               dst_h5[f'{prod_path}/correlationSurfacePeak'],
                               1, offs_params['lines_per_block'])

    t_elapsed = time.time() - t_all
    info_channel.log(
        f"successfully ran offsets product in {t_elapsed:.3f} seconds")

def set_ampcor_params(cfg, ampcor_obj):
    '''
    Set Ampcor optional object parameters
    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined Ampcor parameters
    ampcor_obj: isce3.cuda.matchtemplate.PyCuAmpcor()
        Ampcor object to set members value
    '''

    error_channel = journal.error('offsets_product.set_ampcor_param')

    # Set skip window
    ampcor_obj.skipSampleAcross = cfg['skip_range']
    ampcor_obj.skipSampleDown = cfg['skip_azimuth']

    # Set starting pixel and offset shape
    az_start, rg_start = get_start_pixels(cfg)
    ampcor_obj.referenceStartPixelAcrossStatic = rg_start
    ampcor_obj.referenceStartPixelDownStatic = az_start
    off_length, off_width = get_offsets_shape(cfg,
                                              ampcor_obj.referenceImageHeight,
                                              ampcor_obj.referenceImageWidth)
    ampcor_obj.numberWindowAcross = off_width
    ampcor_obj.numberWindowDown = off_length

    # Set cross-correlation domain, oversampling factor and deramping
    ampcor_obj.algorithm = 0 if cfg['cross_correlation_domain'] == \
                                'frequency' else 1
    ampcor_obj.rawDataOversamplingFactor = cfg['slc_oversampling_factor']
    ampcor_obj.derampMethod = 0 if cfg['deramping_method'] == \
                                   'magnitude' else 1
    ampcor_obj.corrStatWindowSize = cfg['correlation_statistics_zoom']
    ampcor_obj.corrSurfaceZoomInWindow = cfg['correlation_surface_zoom']
    ampcor_obj.corrSurfaceOverSamplingFactor = cfg[
        'correlation_surface_oversampling_factor']
    ampcor_obj.corrSurfaceOverSamplingMethod = 0 if \
        cfg['correlation_surface_oversampling_method'] == 'fft' else 1
    ampcor_obj.numberWindowAcrossInChunk = cfg['windows_batch_range']
    ampcor_obj.numberWindowDownInChunk = cfg['windows_batch_azimuth']
    ampcor_obj.nStreams = cfg['cuda_streams']

    # Setup object parameters and check gross/variable dense offsets
    ampcor_obj.setupParams()
    ampcor_obj.setConstantGrossOffset(cfg['gross_offset_azimuth'],
                                      cfg['gross_offset_range'])

    if cfg['gross_offset_filepath'] is not None:
        gross_offset = np.fromfile(cfg['gross_offset_filepath'], dtype=np.int32)
        windows_number = ampcor_obj.numberWindowAcross * ampcor_obj.numberWindowDown
        if gross_offset.size != 2 * windows_number:
            err_str = "The input gross offset does not match the offsets width*length"
            error_channel.log(err_str)
            raise RuntimeError(err_str)
        gross_offset = gross_offset.reshape(windows_number, 2)
        gross_azimuth = gross_offset[:, 0]
        gross_range = gross_offset[:, 1]
        ampcor_obj.setVaryingGrossOffset(gross_azimuth, gross_range)
    ampcor_obj.mergeGrossOffset = cfg['merge_gross_offset']

    # Check pixel in image range
    ampcor_obj.checkPixelInImageRange()

    return ampcor_obj


def get_offsets_shape(cfg, slc_lines, slc_cols):
    '''
    Get common offset shape among offset layers
    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters for offset layers
    slc_lines: int
        Number of lines of reference RSLC
    slc_cols: int
        Number of columns of reference RSLC
    Returns
    -------
    off_length: int
        Common length among offsets layers
    off_width: int
        Common width among offsets layers
    '''
    # Compute margin around reference RSLC edges
    margin = max(cfg['margin'], np.abs(cfg['gross_offset_range']),
                 np.abs(cfg['gross_offset_azimuth']))
    off_length = cfg.get('offset_length', None)
    off_width = cfg.get('offset_width', None)

    # If offset length is not assigned, compute a common one
    if off_length is None:
        az_search = [cfg[key].get('half_search_azimuth', None) for key
                     in cfg if key.startswith('layer')]
        az_search = min(list(filter(None, az_search)))
        az_chip = [cfg[key].get('window_azimuth', None) for key
                   in cfg if key.startswith('layer')]
        az_chip = min(list(filter(None, az_chip)))
        margin_az = 2 * margin + 2 * az_search + az_chip
        off_length = (slc_lines - margin_az) // cfg['skip_azimuth']

    # If off_width is not assigned, compute a common one
    if off_width is None:
        rg_search = [cfg[key].get('half_search_range', None) for key
                     in cfg if key.startswith('layer')]
        rg_search = min(list(filter(None, rg_search)))
        rg_chip = [cfg[key].get('window_range', None) for key
                   in cfg if key.startswith('layer')]
        rg_chip = min(list(filter(None, rg_chip)))
        margin_rg = 2 * margin + 2 * rg_search + rg_chip
        off_width = (slc_cols - margin_rg) // cfg['skip_range']

    return off_length, off_width


def get_start_pixels(cfg):
    '''
    Get common start pixel among offset layers
    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined parameters for offsets layers
    Returns
    -------
    az_start, rg_start: int
        Start pixel in ref RSLC in azimuth and slant range directions
    '''
    # Compute margin around reference RSLC edges
    margin = max(cfg['margin'], np.abs(cfg['gross_offset_range']),
                 np.abs(cfg['gross_offset_azimuth']))
    rg_start = cfg['start_pixel_range']
    az_start = cfg['start_pixel_azimuth']

    # If None, compute the default start pixel
    if rg_start is None:
        rg_search = [cfg[key].get('half_search_range', None) for key
                     in cfg if key.startswith('layer')]
        rg_start = margin + min(list(filter(None, rg_search)))

    if az_start is None:
        az_search = [cfg[key].get('half_search_azimuth', None) for key
                     in cfg if key.startswith('layer')]
        az_start = margin + min(list(filter(None, az_search)))

    return az_start, rg_start

def write_along_track_offsets_data(infile, dst_h5_ds, band, lines_per_block,
                                   ground_track_velocity_file,
                                   ref_zero_doppler_time_spacing):
    """
    Write along track offsets data from GDAL raster to HDF5 layer

    Parameters
    ----------
    infile: str
        File path to GDAL-friendly raster from where to read data
    dst_h5_ds: h5py.Dataset
        h5py Dataset where to write the data
    band: int
        Band of infile to read data from
    lines_per_block: int
        Lines per block to read in batch
    ground_track_velocity_file: str
        GDAL-friendly file path of ground track velocity of the radargrid
        generated by the get_geometry_product
    ref_zero_doppler_time_spacing : float
        Zero doppler time spacing of the reference RSLC
    """
    # Get shape of input file (same as output created from prep_insar)
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    # Open the ground track velocity file
    ground_track_velocity_ds = gdal.Open(ground_track_velocity_file,
                                         gdal.GA_ReadOnly)

    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    # Iterate over available number of blocks
    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = length - line_start
        else:
            block_length = lines_per_block

        # Get along track offsets, convert to meters, and write to dataset
        # Read in along track offsets as pixels
        ground_track_velocity_data_block = \
            ground_track_velocity_ds.\
                GetRasterBand(1).ReadAsArray(0,
                                             line_start,
                                             width,
                                             block_length)

        data_block = ds.GetRasterBand(band).ReadAsArray(0,
                                                        line_start,
                                                        width,
                                                        block_length)

        # Convert the along track pixel offsets to meters using the equation
        # along_track_offset_in_meters =
        # along_track_offset_in_pixels * ground_track_velocity * zero_doppler_spacing_of_RSLC
        data_block *= ground_track_velocity_data_block \
            * ref_zero_doppler_time_spacing

        dst_h5_ds.write_direct(data_block, dest_sel=
        np.s_[line_start:line_start + block_length, :])

    # Add stats to the along track offsets dataset
    # Try the following codes:
    # ds_as_raster = isce3.io.Raster(f"IH5::ID={dst_h5_ds.id.id}".encode("utf-8"),
    #                                update=True)
    # compute_stats_real_data(ds_as_raster, dst_h5_ds)
    # but failed with this error
    # 'function isce3::io::Raster::Raster(const string&, GDALAccess):
    #   failed to create GDAL dataset from file 'IH5::ID=360287970189643682''
    # Therefore, an independent function compute_stats_real_hdf5_dataset is applied here.
    compute_stats_real_hdf5_dataset(dst_h5_ds)


def write_slant_range_offsets_data(infile, dst_h5_ds, band,
                                   lines_per_block,
                                   slant_range_spacing):
    '''
    Write slant range offsets data from GDAL raster to HDF5 layer as meters
    Parameters
    ----------
    infile: str
        File path to GDAL-friendly raster from where read data
    dst_h5_ds: h5py.Dataset
        h5py Dataset where to write the data
    band: int
        Band of infile to read data from
    lines_per_block: int
        Lines per block to read in batch
    slant_range_spacing: float
        Slant Range Spacing of the reference RSLC
    '''
    # Get shape of input file (same as output created from prep_insar)
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    # Iterate over available number of blocks
    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = length - line_start
        else:
            block_length = lines_per_block

        # Get range offsets, convert to meters, and write to dataset
        data_block = ds.GetRasterBand(band).ReadAsArray(0,
                                                        line_start,
                                                        width,
                                                        block_length)
        data_block *= slant_range_spacing
        dst_h5_ds.write_direct(data_block, dest_sel=
        np.s_[line_start:line_start + block_length, :])

    # Add stats to the along track offsets dataset
    # Try the following codes:
    # ds_as_raster = isce3.io.Raster(f"IH5::ID={dst_h5_ds.id.id}".encode("utf-8"),
    #                                update=True)
    # compute_stats_real_data(ds_as_raster, dst_h5_ds)
    # but failed with this error
    # 'function isce3::io::Raster::Raster(const string&, GDALAccess):
    #   failed to create GDAL dataset from file 'IH5::ID=360287970189643682''
    # Therefore, an independent function compute_stats_real_hdf5_dataset is applied here.
    compute_stats_real_hdf5_dataset(dst_h5_ds)


def _write_offsets_covariance_data(infile,
                                   along_track_cov_ds,
                                   slant_range_cov_ds,
                                   cross_cov_ds,
                                   lines_per_block,
                                   ground_track_velocity_file,
                                   ref_zero_doppler_time_spacing,
                                   ref_slant_range_spacing):
    """
    Write offsets covariance data from GDAL raster to HDF5 layer,
    and convert it to meters

    Parameters
    ----------
    infile: str
        File path to GDAL-friendly raster from where read data
    along_track_cov_ds: h5py.Dataset
        The variance of the along track dataset
    slant_range_cov_ds: h5py.Dataset
        The variane of the slant range dataset
    cross_cov_ds : h5py.Dataset
        The covariance between the along track and slant range dataset
    lines_per_block: int
        Lines per block to read in batch
    ground_track_velocity_file: str
        Ground track velocity file in radargrid generated by the get_geometry_product
    ref_zero_doppler_time_spacing : float
        Zero doppler time spacing of the reference RSLC
    ref_slant_range_spacing: float
        Slant range spacing of the reference RSLC
    """

    # Get shape of input file (same as output created from prep_insar)
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    # Open the ground track velocity file
    ground_track_velocity_ds = gdal.Open(ground_track_velocity_file,
                                         gdal.GA_ReadOnly)

    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    # Iterate over available number of blocks
    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = length - line_start
        else:
            block_length = lines_per_block

        # Read ground track velocity data block
        ground_track_velocity_data_block = \
            ground_track_velocity_ds.\
                GetRasterBand(1).ReadAsArray(0,
                                             line_start,
                                             width,
                                             block_length)

        # Read covariance data block along the track, the slant range,
        # and between along track and slant range
        along_track_cov_data_block, slant_range_cov_data_block,\
            cross_cov_data_block = [ds.GetRasterBand(band).\
                ReadAsArray(0, line_start, width, block_length)
                for band in [1, 2, 3]]

        output_slice = np.s_[line_start:line_start + block_length, :]
        # Convert the along track pixel offsets covariance to meters using the equation
        # along_track_offset_covaraince_in_meters =
        # along_track_offset_covaraince_in_pixels *
        # (ground_track_velocity * zero_doppler_spacing_of_RSLC)^2
        along_track_cov_data_block *= (ground_track_velocity_data_block \
            * ref_zero_doppler_time_spacing)**2
        along_track_cov_ds.write_direct(along_track_cov_data_block,
                                        dest_sel=output_slice)

        # Convert the slant range pixel offsets covariance to meters^2 using the equation
        # slant_range_offset_covaraince_in_meters =
        # slant_range_offset_covaraince_in_pixels *
        # slant_range_spacing^2
        slant_range_cov_data_block *= ref_slant_range_spacing**2
        slant_range_cov_ds.write_direct(slant_range_cov_data_block,
                                        dest_sel=output_slice)

        # Convert the cross covriance pixel offsets covariance to meters^2 using the equation
        # cross_offset_covaraince_in_meters =
        # cross_offset_covaraince_in_pixels *
        # slant_range_spacing * ground_track_velocity * zero_doppler_spacing_of_RSLC
        cross_cov_data_block *= ground_track_velocity_data_block \
            * ref_zero_doppler_time_spacing * ref_slant_range_spacing
        cross_cov_ds.write_direct(cross_cov_data_block,
                                  dest_sel=output_slice)

    # Add stats to the covaraince matrix
    for h5_ds in [along_track_cov_ds,
                  slant_range_cov_ds,
                  cross_cov_ds]:
        compute_stats_real_hdf5_dataset(h5_ds)


def write_data(infile, outfile, band, lines_per_block):
    '''
    Write data from GDAL raster to HDF5 layer
    Parameters
    ----------
    infile: str
        File path to GDAL-friendly raster from where read data
    outfile: h5py.File
        h5py Dataset where to write the data
    band: int
        Band of infile to read data from
    lines_per_block: int
        Lines per block to read in batch
    '''
    # Get shape of input file (same as output created from prepare_insar_hdf5)
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    # Iterate over available number of blocks
    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = length - line_start
        else:
            block_length = lines_per_block

        data_block = ds.GetRasterBand(band).ReadAsArray(0,
                                                        line_start,
                                                        width,
                                                        block_length)
        outfile.write_direct(data_block, dest_sel=
        np.s_[line_start:line_start + block_length, :])

    # Add statistics
    raster = isce3.io.Raster(infile)
    compute_stats_real_data(raster, outfile)


if __name__ == "__main__":
    '''Run offset product generation '''
    # Load command line args
    offsets_parser = YamlArgparse()
    args = offsets_parser.parse()

    # Get dictionary from CLI arguments
    offsets_runconfig = OffsetsProductRunConfig(args)

    # Prepare ROFF HDF5 product
    out_paths = prepare_insar_hdf5.run(offsets_runconfig.cfg)

    # Run offsets product generation
    run(offsets_runconfig.cfg, out_paths['ROFF'])
