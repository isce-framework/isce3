'''
Wrapper for dense offsets
'''

import pathlib
import time

import journal
import numpy as np
import isce3
import pycuampcor
from osgeo import gdal
from nisar.products.readers import SLC
from nisar.workflows.helpers import copy_raster, get_cfg_freq_pols
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.dense_offsets_runconfig import \
    DenseOffsetsRunConfig


def get_ampcor_slc(hdf5_file, freq, pol, lines_per_block, copy_path):
    '''
    Get the RSLC image for Ampcor: the HDF5 dataset if pycuampcor can read it
    directly, or else a memory mappable (ENVI) copy

    Parameters
    ----------
    hdf5_file: str
        Path to the RSLC HDF5 file
    freq: str
        Frequency band ('A' or 'B')
    pol: str
        Polarization
    lines_per_block: int
        Lines per block to copy the RSLC dataset
    copy_path: str
        Path to the memory mappable copy, if needed

    Returns
    -------
    str
        The image name for Ampcor: HDF5:<file>:<dataset>, or copy_path
    '''
    slc = SLC(hdf5file=hdf5_file)
    # pycuampcor reads float32 (complex64) datasets, not float16 (complex32)
    if getattr(pycuampcor, 'has_hdf5', False) and \
            not slc.is_dataset_complex32(freq, pol):
        return f'HDF5:{hdf5_file}:/{slc.slcPath(freq, pol)}'
    copy_raster(hdf5_file, freq, pol, lines_per_block, copy_path,
                file_type='ENVI')
    return copy_path


def run(cfg: dict):
    '''
    Run dense offsets
    '''

    # Pull parameters from cfg
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    offset_params = cfg['processing']['dense_offsets']

    # Initialize parameters shared between frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)

    # Get coregistered SLC path
    coregistered_slc_path = pathlib.Path(offset_params['coregistered_slc_path'])

    error_channel = journal.error('dense_offsets.run')
    info_channel = journal.info('dense_offsets.run')
    info_channel.log('Start dense offsets estimation')

    # Check GPU use
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])

    if use_gpu:
        # Set current CUDA device
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)
        ampcor = pycuampcor.PyCuAmpcor()
        ampcor.deviceID = cfg['worker']['gpu_id']
    else:
        ampcor = pycuampcor.PyCPUAmpcor()

    # Use memory mapping (not exposed to user but reference
    # and secondary raster are memory-mappable)
    ampcor.useMmap = 1

    # Looping over frequencies and polarizations
    t_all = time.time()

    for freq, _, pol_list in get_cfg_freq_pols(cfg):
        offset_scratch = scratch_path / f'dense_offsets/freq{freq}'

        for pol in pol_list:
            # Set output directory and output filenames
            out_dir = offset_scratch / pol
            out_dir.mkdir(parents=True, exist_ok=True)

            # Reference SLC: the HDF5 dataset, or a memory mappable copy
            ref_raster_str = f'HDF5:{ref_hdf5}:/{ref_slc.slcPath(freq, pol)}'
            ref_raster = isce3.io.Raster(ref_raster_str)
            ampcor.referenceImageName = get_ampcor_slc(
                ref_hdf5, freq, pol, offset_params['lines_per_block'],
                str(out_dir / 'reference.slc'))
            ampcor.referenceImageHeight = ref_raster.length
            ampcor.referenceImageWidth = ref_raster.width

            # If running insar.py, a memory mappable second raster has been
            # created in the previous step (resample slc). If secondary raster
            # is extracted from HDF5 file, read the HDF5 dataset directly or
            # make a memory mappable copy
            if coregistered_slc_path.is_file():
                sec_raster_path = get_ampcor_slc(
                    sec_hdf5, freq, pol, offset_params['lines_per_block'],
                    str(out_dir / 'secondary.slc'))
            else:
                 sec_raster_path = str(coregistered_slc_path /
                                       f'coarse_resample_slc/freq{freq}/{pol}/coregistered_secondary.slc')
            sec_raster = isce3.io.Raster(sec_raster_path)
            ampcor.secondaryImageName = sec_raster_path
            ampcor.secondaryImageHeight = sec_raster.length
            ampcor.secondaryImageWidth = sec_raster.width

            # Setup other dense offsets parameters
            ampcor = set_optional_attributes(ampcor, offset_params,
                                             ref_raster.length,
                                             ref_raster.width)
            # Configure output filenames. It is assumed output are flat binaries (e.g. ENVI files)
            ampcor.offsetImageName = str(out_dir / 'dense_offsets')
            ampcor.grossOffsetImageName = str(out_dir / 'gross_offset')
            ampcor.snrImageName = str(out_dir / 'snr')
            ampcor.covImageName = str(out_dir / 'covariance')
            ampcor.peakValueImageName = str(out_dir / 'correlation_peak')

            # Create empty ENVI datasets. PyCuAmpcor will overwrite the
            # binary files. Note, use gdal to pass interleave option
            create_empty_dataset(str(out_dir / 'dense_offsets'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 2, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'gross_offsets'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 2, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'snr'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 1, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'covariance'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 3, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'correlation_peak'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 1, gdal.GDT_Float32)
            # Run dense offsets
            ampcor.runAmpcor()

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"Successfully ran dense_offsets in {t_all_elapsed:.3f} seconds")


def set_optional_attributes(ampcor_obj, cfg, length, width):
    '''
    Set obj attributes to cfg values
    Check attributes validity
    '''

    error_channel = journal.error('dense_offsets.run.set_optional_attribute')
    if cfg['window_range'] is not None:
        ampcor_obj.windowSizeWidth = cfg['window_range']

    if cfg['window_azimuth'] is not None:
        ampcor_obj.windowSizeHeight = cfg['window_azimuth']

    if cfg['half_search_range'] is not None:
        ampcor_obj.halfSearchRangeAcross = cfg['half_search_range']

    if cfg['half_search_azimuth'] is not None:
        ampcor_obj.halfSearchRangeDown = cfg['half_search_azimuth']

    if cfg['skip_range'] is not None:
        ampcor_obj.skipSampleAcross = cfg['skip_range']

    if cfg['skip_azimuth'] is not None:
        ampcor_obj.skipSampleDown = cfg['skip_azimuth']

    if cfg['margin'] is not None:
        margin = cfg['margin']
    else:
        margin = 0

    # If gross offsets are set update margin
    if (cfg['gross_offset_range'] is not None) and (cfg['gross_offset_azimuth'] is not None):
        margin = max(margin, np.abs(cfg['gross_offset_range']),
                     np.abs(cfg['gross_offset_azimuth']))

    margin_rg = 2 * margin + 2*ampcor_obj.halfSearchRangeAcross + ampcor_obj.windowSizeWidth
    margin_az = 2 * margin + 2*ampcor_obj.halfSearchRangeDown + ampcor_obj.windowSizeHeight

    ampcor_obj.referenceStartPixelAcrossStatic = cfg[
        'start_pixel_range'] if cfg['start_pixel_range'] is not None \
        else margin + ampcor_obj.halfSearchRangeAcross

    ampcor_obj.referenceStartPixelDownStatic = cfg[
        'start_pixel_azimuth'] if cfg['start_pixel_azimuth'] is not None \
        else margin + ampcor_obj.halfSearchRangeDown

    if cfg['offset_width'] is not None:
        ampcor_obj.numberWindowAcross = cfg['offset_width']
    else:
        offset_width = (width - margin_rg) // ampcor_obj.skipSampleAcross
        if offset_width <= 0:
            err_str = (f"Image width ({width}) is too small for the configured parameters "
                       f"(margin_rg={margin_rg}): no valid ampcor windows fit across range.")
            error_channel.log(err_str)
            raise ValueError(err_str)
        ampcor_obj.numberWindowAcross = offset_width

    if cfg['offset_length'] is not None:
        ampcor_obj.numberWindowDown = cfg['offset_length']
    else:
        offset_length = (length - margin_az) // ampcor_obj.skipSampleDown
        if offset_length <= 0:
            err_str = (f"Image length ({length}) is too small for the configured parameters "
                       f"(margin_az={margin_az}): no valid ampcor windows fit along azimuth.")
            error_channel.log(err_str)
            raise ValueError(err_str)
        ampcor_obj.numberWindowDown = offset_length

    if cfg['cross_correlation_domain'] is not None:
        algorithm = cfg['cross_correlation_domain']
        if algorithm == 'frequency':
            ampcor_obj.algorithm = 0
        elif algorithm == 'spatial':
            ampcor_obj.algorithm = 1
        else:
            err_str = f"{algorithm} is not a valid cross-correlation option"
            error_channel.log(err_str)
            raise ValueError(err_str)

    if cfg.get('cross_correlation_workflow') is not None:
        ampcor_obj.workflow = get_ampcor_workflow(cfg['cross_correlation_workflow'])

    if cfg['slc_oversampling_factor'] is not None:
        ampcor_obj.rawDataOversamplingFactor = cfg['slc_oversampling_factor']

    if cfg['deramping_method'] is not None:
        deramp = cfg['deramping_method']
        if deramp == "magnitude":
            ampcor_obj.derampMethod = 0
        elif deramp == "complex":
            ampcor_obj.derampMethod = 1
        else: # skip deramping
            ampcor_obj.derampMethod = 2

    if cfg['deramping_axis'] is not None:
        deramp_axis = cfg['deramping_axis']
        if deramp_axis == "azimuth":
            ampcor_obj.derampAxis = 0
        elif deramp_axis == "range":
            ampcor_obj.derampAxis = 1
        else: # both directions
            ampcor_obj.derampAxis = 2

    if cfg['correlation_statistics_zoom'] is not None:
        ampcor_obj.corrStatWindowSize = cfg['correlation_statistics_zoom']

    if cfg['correlation_surface_zoom'] is not None:
        ampcor_obj.corrSurfaceZoomInWindow = cfg['correlation_surface_zoom']

    if cfg['correlation_surface_oversampling_factor'] is not None:
        ampcor_obj.corrSurfaceOverSamplingFactor = cfg[
            'correlation_surface_oversampling_factor']

    if cfg['correlation_surface_oversampling_method'] is not None:
        method = cfg['correlation_surface_oversampling_method']
        ampcor_obj.corrSurfaceOverSamplingMethod = 0 if method == "fft" else 1

    if cfg['windows_batch_range'] is not None:
        ampcor_obj.numberWindowAcrossInChunk = cfg['windows_batch_range']

    if cfg['windows_batch_azimuth'] is not None:
        ampcor_obj.numberWindowDownInChunk = cfg['windows_batch_azimuth']

    if cfg['cuda_streams'] is not None:
        ampcor_obj.nStreams = cfg['cuda_streams']

    # Setup object parameters
    ampcor_obj.setupParams()
    if (cfg['use_gross_offsets'] is not None) and (
            cfg['gross_offset_range'] is not None) and \
            (cfg['gross_offset_azimuth'] is not None):
        ampcor_obj.setConstantGrossOffset(cfg['gross_offset_azimuth'],
                                          cfg['gross_offset_range'])

    if cfg['gross_offset_filepath'] is not None:
        gross_offset = np.fromfile(cfg['gross_offset_filepath'], dtype=np.int32)
        windows_number = ampcor_obj.numberWindowAcross * ampcor_obj.numberWindowDown
        if gross_offset.size != 2 * windows_number:
            err_str = "The input gross offset does not match the offset width*offset length"
            error_channel.log(err_str)
            raise RuntimeError(err_str)
        gross_offset = gross_offset.reshape(windows_number, 2)
        gross_azimuth = gross_offset[:, 0]
        gross_range = gross_offset[:, 1]
        ampcor_obj.setVaryingGrossOffset(gross_azimuth, gross_range)

    # If True, add constant slant range/azimuth gross offsets to
    # estimated dense offsets (will be used to resample slc)
    if cfg['merge_gross_offset'] is not None:
        ampcor_obj.mergeGrossOffset = 1 if cfg['merge_gross_offset'] else 0

    # Check pixel in image range; warns to stderr when out of range but does not abort
    ampcor_obj.checkPixelInImageRange()

    return ampcor_obj


def get_ampcor_workflow(workflow):
    '''
    Convert the cross-correlation workflow name to the pycuampcor option

    Parameters
    ----------
    workflow: str
        'two_pass': a first pass without anti-aliasing oversampling to
        estimate the pixel-level offsets, and a second pass with
        oversampling over a smaller search range; or
        'one_pass': a single pass with anti-aliasing oversampling over the
        whole search range (more accurate for noisy correlation surfaces)

    Returns
    -------
    int
        0 for 'two_pass', 1 for 'one_pass'
    '''
    workflows = {'two_pass': 0, 'one_pass': 1}
    if workflow not in workflows:
        err_str = f"{workflow} is not a valid cross-correlation workflow"
        journal.error('dense_offsets.get_ampcor_workflow').log(err_str)
        raise ValueError(err_str)
    return workflows[workflow]


def create_empty_dataset(filename, width, length,
                         bands, dtype, interleave="bip", file_type="ENVI"):
    '''
    Create empty dataset with user-defined options
    '''
    driver = gdal.GetDriverByName(file_type)
    driver.Create(filename, xsize=width, ysize=length, bands=bands,
                  eType=dtype, options=[f"INTERLEAVE={interleave}"])


if __name__ == "__main__":
    '''
    Run dense offsets estimation
    '''
    # Load command line args
    dense_offsets_parser = YamlArgparse()
    args = dense_offsets_parser.parse()
    # Get cfg dict from CLI args
    dense_offsets_runconfig = DenseOffsetsRunConfig(args)
    # Run dense offsets
    run(dense_offsets_runconfig.cfg)
