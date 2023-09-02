'''
Wrapper for dense offsets
'''

import pathlib
import time

import journal
import numpy as np
import isce3
from osgeo import gdal
from nisar.products.readers import SLC
from nisar.workflows.helpers import copy_raster, get_cfg_freq_pols
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.dense_offsets_runconfig import \
    DenseOffsetsRunConfig


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
        ampcor = isce3.cuda.matchtemplate.PyCuAmpcor()
        ampcor.deviceID = cfg['worker']['gpu_id']
    else:
        ampcor = isce3.matchtemplate.PyCPUAmpcor()

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

            # Create a memory mappable copy of reference SLC
            copy_raster(ref_hdf5, freq, pol,
                        offset_params['lines_per_block'],
                        str(out_dir / 'reference.slc'), file_type='ENVI')

            ref_raster_str = f'HDF5:{ref_hdf5}:/{ref_slc.slcPath(freq, pol)}'
            ref_raster = isce3.io.Raster(ref_raster_str)
            ampcor.referenceImageName = str(out_dir / 'reference.slc')
            ampcor.referenceImageHeight = ref_raster.length
            ampcor.referenceImageWidth = ref_raster.width

            # If running insar.py, a memory mappable second raster has been
            # created in the previous step (resample slc). If secondary raster
            # is extracted from HDF5 file, needs to be made memory mappable
            if coregistered_slc_path.is_file():
                sec_raster_path = str(out_dir / 'secondary.slc')
                copy_raster(sec_hdf5, freq, pol,
                            offset_params['lines_per_block'],
                            sec_raster_path, file_type='ENVI')
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
            ampcor.corrImageName = str(out_dir / 'correlation_peak')

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
        'start_pixel_azimuth'] if cfg['start_pixel_range'] is not None \
        else margin + ampcor_obj.halfSearchRangeDown

    if cfg['offset_width'] is not None:
        ampcor_obj.numberWindowAcross = cfg['offset_width']
    else:
        offset_width = (width - margin_rg) // ampcor_obj.skipSampleAcross
        ampcor_obj.numberWindowAcross = offset_width

    if cfg['offset_length'] is not None:
        ampcor_obj.numberWindowDown = cfg['offset_length']
    else:
        offset_length = (length - margin_az) // ampcor_obj.skipSampleDown
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

    if cfg['slc_oversampling_factor'] is not None:
        ampcor_obj.rawDataOversamplingFactor = cfg['slc_oversampling_factor']

    if cfg['deramping_method'] is not None:
        deramp = cfg['deramping_method']
        ampcor_obj.derampMethod = 0 if deramp == "magnitude" else 1

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

    # Check pixel in image range
    ampcor_obj.checkPixelInImageRange()

    return ampcor_obj


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
