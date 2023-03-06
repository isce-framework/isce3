import pathlib
import time

import h5py
import isce3
import journal
import numpy as np
from nisar.products.readers import SLC
from nisar.workflows import h5_prep
from nisar.workflows.dense_offsets import create_empty_dataset
from nisar.workflows.helpers import copy_raster
from nisar.workflows.offsets_product_runconfig import OffsetsProductRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.compute_stats import compute_stats_real_data
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
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    offs_params = cfg['processing']['offsets_product']
    coreg_slc_path = pathlib.Path(offs_params['coregistered_slc_path'])

    # Initialize parameters shared between frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

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

    # Loop over frequencies and polarizations
    t_all = time.time()
    with h5py.File(output_hdf5, 'a', libver='latest', swmr=True) as dst_h5:
        for freq, pol_list in freq_pols.items():
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
                    lay_scratch = out_dir / key
                    lay_scratch.mkdir(parents=True, exist_ok=True)
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
                            lay_scratch / 'dense_offsets')
                    ampcor.grossOffsetImageName = str(
                            lay_scratch / 'gross_offset')
                    ampcor.snrImageName = str(lay_scratch / 'snr')
                    ampcor.covImageName = str(lay_scratch / 'covariance')
                    create_empty_dataset(str(lay_scratch / 'dense_offsets'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 2,
                                         gdal.GDT_Float32)
                    create_empty_dataset(str(lay_scratch / 'gross_offsets'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 2,
                                         gdal.GDT_Float32)
                    create_empty_dataset(str(lay_scratch / 'snr'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 1,
                                         gdal.GDT_Float32)
                    create_empty_dataset(str(lay_scratch / 'covariance'),
                                         ampcor.numberWindowAcross,
                                         ampcor.numberWindowDown, 3,
                                         gdal.GDT_Float32)

                    # Run ampcor and delete ampcor object after is done
                    ampcor.runAmpcor()
                    del ampcor

                    # Write Ampcor datasets in HDF5 ROFF product
                    prod_path = f'science/LSAR/ROFF/swaths/' \
                                f'frequency{freq}/pixelOffsets/{pol}/{key}'

                    # Write datasets
                    write_data(str(lay_scratch / 'dense_offsets'),
                               dst_h5[f'{prod_path}/alongTrackOffset'],
                               1, offs_params['lines_per_block'])
                    write_data(str(lay_scratch / 'dense_offsets'),
                               dst_h5[f'{prod_path}/slantRangeOffset'],
                               2, offs_params['lines_per_block'])
                    write_data(str(lay_scratch / 'covariance'),
                               dst_h5[f'{prod_path}/alongTrackOffsetVariance'],
                               1, offs_params['lines_per_block'])
                    write_data(str(lay_scratch / 'covariance'),
                               dst_h5[f'{prod_path}/slantRangeOffsetVariance'],
                               2, offs_params['lines_per_block'])
                    write_data(str(lay_scratch / 'covariance'),
                               dst_h5[f'{prod_path}/crossOffsetVariance'],
                               3, offs_params['lines_per_block'])
                    write_data(str(lay_scratch / 'snr'),
                               dst_h5[f'{prod_path}/snr'],
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
    # Get shape of input file (same as output created from h5_prep)
    ds = gdal.Open(infile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    # Iterate over available number of block
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
    out_paths = h5_prep.run(offsets_runconfig.cfg)

    # Run offsets product generation
    run(offsets_runconfig.cfg, out_paths['ROFF'])
