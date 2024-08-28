#!/usr/bin/env python3

"""
Wrapper for phase unwrapping
"""

import pathlib
import time

import isce3
import journal
import numpy as np
import snaphu
from isce3.core import crop_external_orbit
from isce3.io import HDF5OptimizedReader
from isce3.unwrap.preprocess import preprocess_wrapped_igram as preprocess
from isce3.unwrap.preprocess import project_map_to_radar
from nisar.products.insar.product_paths import RIFGGroupsPaths
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows import crossmul, prepare_insar_hdf5
from nisar.workflows.compute_stats import (compute_stats_real_data,
                                           compute_stats_real_hdf5_dataset)
from nisar.workflows.helpers import get_cfg_freq_pols
from nisar.workflows.unwrap_runconfig import UnwrapRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from osgeo import gdal


def run(cfg: dict, input_hdf5: str, output_hdf5: str):
    """
    run phase unwrapping

    Parameters
    ---------
    cfg: dict
        Dictionary with user-defined options
    input_hdf5: str
        File path to input HDF5 product (i.e., RIFG)
    output_hdf5: str
        File path to output HDF5 product (i.e., RUNW)
    """

    # pull parameters from dictionary
    ref_slc_hdf5 = cfg['input_file_group']['reference_rslc_file']
    ref_orbit_ext = cfg['dynamic_ancillary_file_group']['orbit_files']['reference_orbit_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    unwrap_args = cfg['processing']['phase_unwrap']
    unwrap_rg_looks = cfg['processing']['phase_unwrap']['range_looks']
    unwrap_az_looks = cfg['processing']['phase_unwrap']['azimuth_looks']

    # Instantiate RIFG obj to avoid hard-coded paths to RIFG datasets
    rifg_obj = RIFGGroupsPaths()

    # Create error and info channels
    error_channel = journal.error("unwrap.run")
    info_channel = journal.info("unwrap.run")
    info_channel.log("Starting phase unwrapping")

    crossmul_path = pathlib.Path(input_hdf5)
    # if not file or directory raise error
    if not crossmul_path.is_file():
        err_str = f"{crossmul_path} is invalid; needs to be a file"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Open reference RSLC object
    ref_slc = SLC(hdf5file=ref_slc_hdf5)

    # Start to track time
    t_all = time.time()

    with HDF5OptimizedReader(name=output_hdf5, mode="a", libver="latest", swmr=True) as dst_h5, \
            HDF5OptimizedReader(name=crossmul_path, mode="r", libver="latest", swmr=True) as src_h5:
        for freq, pol_list, offset_pol_list in get_cfg_freq_pols(cfg):
            src_freq_group_path = f"{rifg_obj.SwathsPath}/frequency{freq}"
            src_freq_bandwidth_group_path = (f"{rifg_obj.ProcessingInformationPath}/parameters"
                                             f"/reference/frequency{freq}")
            dst_freq_group_path = src_freq_group_path.replace("RIFG", "RUNW")

            # Get reference RSLC orbit object
            ref_orbit = ref_slc.getOrbit()
            if ref_orbit_ext is not None:
                external_orbit = load_orbit_from_xml(ref_orbit_ext,
                                                     ref_slc.getRadarGrid(freq).ref_epoch)
                ref_orbit = crop_external_orbit(external_orbit, ref_orbit)

            for pol in pol_list:
                src_pol_group_path = \
                    f"{src_freq_group_path}/interferogram/{pol}"
                dst_pol_group_path = \
                    f"{dst_freq_group_path}/interferogram/{pol}"

                # Fetch paths to input/output datasets
                igram_path = \
                    f"HDF5:{crossmul_path}:/{src_pol_group_path}/wrappedInterferogram"
                corr_path = \
                    f"HDF5:{crossmul_path}:/{src_pol_group_path}/coherenceMagnitude"

                # Create unwrapped interferogram output raster
                unw_path = f"{dst_pol_group_path}/unwrappedPhase"
                unw_dataset = dst_h5[unw_path]
                unw_raster_path = \
                    f"IH5:::ID={unw_dataset.id.id}".encode("utf-8")

                # Create connected components output raster
                conn_comp_path = f"{dst_pol_group_path}/connectedComponents"
                conn_comp_dataset = dst_h5[conn_comp_path]
                conn_comp_raster_path = \
                    f"IH5:::ID={conn_comp_dataset.id.id}".encode("utf-8")

                # Create unwrapping scratch directory to store temporary rasters
                crossmul_scratch = scratch_path / f'crossmul/freq{freq}/{pol}/'
                unwrap_scratch = scratch_path / f'unwrap/freq{freq}/{pol}'
                unwrap_scratch.mkdir(parents=True, exist_ok=True)

                # If requested, run crossmul with a different number of looks.
                # Use the generated wrapped interferogram and coherence for
                # unwrapping.
                if (unwrap_rg_looks > 1) or (unwrap_az_looks > 1):
                    if cfg['processing']['fine_resample']['enabled']:
                        resample_type = 'fine'
                    else:
                        resample_type = 'coarse'
                    crossmul.run(cfg, output_hdf5=None, resample_type=resample_type,
                                 dump_on_disk=True, rg_looks=unwrap_rg_looks,
                                 az_looks=unwrap_az_looks)
                    igram_path = str(crossmul_scratch / 'wrapped_igram_rg'
                                     f'{unwrap_rg_looks}_az{unwrap_az_looks}')
                    corr_path = str(crossmul_scratch / 'coherence_rg'
                                    f'{unwrap_rg_looks}_az{unwrap_az_looks}')

                # If enabled, preprocess wrapped phase: remove invalid pixels
                # and fill their location with a filling algorithm
                if unwrap_args["preprocess_wrapped_phase"]["enabled"]:
                    # Extract preprocessing dictionary and open arrays
                    preproc_cfg = unwrap_args["preprocess_wrapped_phase"]
                    filling_enabled = preproc_cfg["filling_enabled"]
                    filling_method = preproc_cfg["filling_method"]
                    igram = open_raster(igram_path)
                    coherence = open_raster(corr_path)
                    mask = (
                        open_raster(preproc_cfg["mask"]["mask_path"])
                        if preproc_cfg["mask"]["mask_path"] is not None
                        else None)

                    if "water" in preproc_cfg["mask"]["mask_type"]:
                        # water_mask_file is expected to have distance from the boundary of the
                        # water bodies. The values 0-100 represent the distance from the coastline
                        # and values from 101-200 represent the distance from
                        # inland water boundaries.
                        water_mask_path = \
                            cfg["dynamic_ancillary_file_group"]["water_mask_file"]
                        ocean_water_buffer = \
                            preproc_cfg["mask"]["ocean_water_buffer"]
                        inland_water_buffer = \
                            preproc_cfg["mask"]["inland_water_buffer"]
                        water_distance = project_map_to_radar(
                            cfg, water_mask_path, freq)
                        # Since distance from inland water is defined from 101 to 200 in water mask file,
                        # the value 100 needs to be added.
                        inland_water_mask = water_distance > inland_water_buffer + 100
                        ocean_water_mask = (
                            water_distance > ocean_water_buffer
                        ) & (water_distance <= 100)
                        if mask is not None:
                            mask = mask | inland_water_mask | ocean_water_mask
                        else:
                            mask = inland_water_mask | ocean_water_mask

                    if filling_method == "distance_interpolator":
                        distance = \
                            preproc_cfg["distance_interpolator"]["distance"]

                    igram_filt = preprocess(
                        igram,
                        coherence,
                        mask,
                        preproc_cfg["mask"]["mask_type"],
                        preproc_cfg["mask"]["outlier_threshold"],
                        preproc_cfg["mask"]["median_filter_size"],
                        filling_enabled,
                        filling_method,
                        distance)
                    # Save filtered/filled wrapped interferogram
                    igram_path = f'{unwrap_scratch}/wrapped_igram.filt'
                    write_raster(igram_path, igram_filt)

                # Run unwrapping based on user-defined algorithm
                algorithm = unwrap_args["algorithm"]

                if algorithm == "icu":
                    info_channel.log("Unwrapping with ICU")
                    icu_cfg = unwrap_args["icu"]
                    icu_obj = set_icu_attributes(icu_cfg)

                    # Allocate input/output rasters
                    igram_raster = isce3.io.Raster(igram_path)
                    corr_raster = isce3.io.Raster(corr_path)
                    unw_raster = isce3.io.Raster(unw_raster_path, update=True)
                    conn_comp_raster = isce3.io.Raster(
                        conn_comp_raster_path, update=True)
                    # Run unwrapping
                    icu_obj.unwrap(
                        unw_raster,
                        conn_comp_raster,
                        igram_raster,
                        corr_raster,
                        seed=icu_cfg["seed"])
                    # Compute statistics
                    compute_stats_real_data(unw_raster, unw_dataset)
                    # Log attributes for ICU
                    log_unwrap_attributes(icu_obj, info_channel, algorithm)
                    # Clean connected components raster
                    del conn_comp_raster

                elif algorithm == "phass":
                    info_channel.log("Unwrapping using PHASS")
                    phass_cfg = unwrap_args["phass"]
                    phass_obj = set_phass_attributes(phass_cfg)

                    # Phass requires the phase of igram (not complex igram)
                    # Generate InSAR phase using GDAL pixel functions
                    igram_phase_path = unwrap_scratch / "wrapped_phase.vrt"
                    igram_phase_to_vrt(igram_path, str(igram_phase_path))

                    # Allocate input/output raster
                    igram_phase_raster = isce3.io.Raster(str(igram_phase_path))
                    corr_raster = isce3.io.Raster(corr_path)
                    unw_raster = isce3.io.Raster(unw_raster_path, update=True)
                    conn_comp_raster = isce3.io.Raster(
                        conn_comp_raster_path, update=True)

                    # Check if it is required to unwrap with power raster
                    if phass_cfg.get("power") is not None:
                        power_raster = isce3.io.Raster(phass_cfg["power"])
                        phass_obj.unwrap(
                            igram_phase_raster,
                            power_raster,
                            corr_raster,
                            unw_raster,
                            conn_comp_raster)
                    else:
                        phass_obj.unwrap(
                            igram_phase_raster,
                            corr_raster,
                            unw_raster,
                            conn_comp_raster)
                    # Compute statistics
                    compute_stats_real_data(unw_raster, unw_dataset)
                    # Log attributes for phass
                    log_unwrap_attributes(phass_obj, info_channel, algorithm)
                    # Clean connected components raster
                    del conn_comp_raster
                elif algorithm == "snaphu":
                    info_channel.log("Unwrapping with SNAPHU")

                    # Get SNAPHU dictionary with user params
                    snaphu_cfg = unwrap_args["snaphu"]

                    # Get input array to run unwrapping with snaphu-py
                    igram_array = open_raster(igram_path)
                    coh_array = open_raster(corr_path)

                    mask_array = open_raster(
                        snaphu_cfg['mask']) if snaphu_cfg['mask'] is not None else None

                    # Get effective number of looks
                    if snaphu_cfg['nlooks'] is not None:
                        nlooks = snaphu_cfg['nlooks']
                    else:
                        rg_spacing = src_h5[f"{src_freq_group_path}/interferogram/slantRangeSpacing"][()]
                        az_spacing = src_h5[f"{src_freq_group_path}/interferogram/sceneCenterAlongTrackSpacing"][()]
                        rg_bw = src_h5[f"{src_freq_bandwidth_group_path}/rangeBandwidth"][()]
                        az_bw = src_h5[f"{src_freq_bandwidth_group_path}/azimuthBandwidth"][()]
                        nlooks = get_effective_looks(ref_slc, ref_orbit, rg_spacing,
                                                     az_spacing, rg_bw, az_bw, freq=freq)
                    # Run snaphu using snaphu-py
                    snaphu.unwrap(igram_array, coh_array, nlooks,
                                  unw=dst_h5[unw_path],
                                  conncomp=dst_h5[conn_comp_path],
                                  cost=snaphu_cfg['cost_mode'],
                                  mask=mask_array,
                                  init=snaphu_cfg['initialization_method'],
                                  min_conncomp_frac=snaphu_cfg['min_conncomp_frac'],
                                  phase_grad_window=snaphu_cfg['phase_grad_window'],
                                  ntiles=snaphu_cfg['ntiles'],
                                  tile_overlap=snaphu_cfg['tile_overlap'],
                                  nproc=snaphu_cfg['nproc'],
                                  tile_cost_thresh=snaphu_cfg['tile_cost_thresh'],
                                  min_region_size=snaphu_cfg['min_region_size'],
                                  single_tile_reoptimize=snaphu_cfg['single_tile_reoptimize'],
                                  regrow_conncomps=snaphu_cfg['regrow_conncomps'],
                                  scratchdir=unwrap_scratch,
                                  delete_scratch=True)

                    # Compute statistics (stats module supports isce3.io.Raster)
                    unw_raster = isce3.io.Raster(unw_raster_path)
                    compute_stats_real_data(unw_raster, unw_dataset)

                else:
                    err_str = f"{algorithm} is an invalid unwrapping algorithm"
                    error_channel.log(err_str)

                # Clean up unwrapped phase raster
                del unw_raster
                
                # Allocate coherence in RUNW. If no further multilooking, the coherence
                # is copied from RIFG
                datasets = ['coherenceMagnitude']
                groups = ['interferogram']
                # append datasets/groups for offsets if polarization exists
                # offset polarizations can differ from interferogram
                # polarizations if single pol offset mode enabled
                if pol in offset_pol_list:
                    datasets.extend(['alongTrackOffset',
                                     'slantRangeOffset',
                                     'correlationSurfacePeak'])
                    groups.extend(
                        ['pixelOffsets', 'pixelOffsets', 'pixelOffsets'])
                for dataset, group in zip(datasets, groups):
                    dst_path = f'{dst_freq_group_path}/{group}/{pol}/{dataset}'
                    src_path = f'{src_freq_group_path}/{group}/{pol}/{dataset}'
                    if (dataset == 'coherenceMagnitude') and ((unwrap_rg_looks > 1)
                                                              or (unwrap_az_looks > 1)):
                        corr_path = \
                            str(f'{crossmul_scratch}/coherence_rg{unwrap_rg_looks}_az{unwrap_az_looks}')
                        corr = open_raster(corr_path)
                        dst_h5[dst_path][:, :] = corr
                    else:
                        dst_h5[dst_path][:, :] = src_h5[src_path][()]

                    dst_dataset = dst_h5[dst_path]
                    dst_raster = isce3.io.Raster(
                        f"IH5:::ID={dst_dataset.id.id}".encode("utf-8"),
                        update=True)
                    if dataset not in ['correlationSurfacePeak']:
                        compute_stats_real_data(dst_raster, dst_dataset)


    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"Successfully ran phase unwrapping in {t_all_elapsed:.3f} seconds"
    )


def log_unwrap_attributes(unwrap, info, algorithm):
    """
    Write unwrap attributes to info channel

    Parameters
    ---------
    unwrap: object
        Unwrapping object
    info: journal.info
        Info channel where to log attributes values
    algorithm: str
        String identifying unwrapping algorithm being used
    """
    info.log(f"Unwrapping algorithm:{algorithm}")
    if algorithm == "icu":
        info.log(f"Correlation threshold increments: {unwrap.corr_incr_thr}")
        info.log(f"Number of buffer lines: {unwrap.buffer_lines}")
        info.log(f"Number of overlap lines: {unwrap.overlap_lines}")
        info.log(f"Use phase gradient neutron: {unwrap.use_phase_grad_neut}")
        info.log(f"Use intensity neutron: {unwrap.use_intensity_neut}")
        info.log(f"Phase gradient window size: {unwrap.phase_grad_win_size}")
        info.log(f"Phase gradient threshold: {unwrap.neut_phase_grad_thr}")
        info.log(f"Neutron intensity threshold: {unwrap.neut_intensity_thr}")
        info.log(
            "Maximum intensity correlation "
            f"threshold: {unwrap.neut_correlation_thr}"
        )
        info.log(f"Number of trees: {unwrap.trees_number}")
        info.log(f"Maximum branch length: {unwrap.max_branch_length}")
        info.log(f"Pixel spacing ratio: {unwrap.ratio_dxdy}")
        info.log(f"Initial correlation threshold: {unwrap.init_corr_thr}")
        info.log(f"Maximum correlation threshold: {unwrap.max_corr_thr}")
        info.log(f"Correlation threshold increments: {unwrap.corr_incr_thr}")
        info.log(f"Minimum tile area fraction: {unwrap.min_cc_area}")
        info.log(f"Number of bootstrapping lines: {unwrap.num_bs_lines}")
        info.log(f"Minimum overlapping area: {unwrap.min_overlap_area}")
        info.log(f"Phase variance threshold: {unwrap.phase_var_thr}")
    elif algorithm == "phass":
        info.log(
            f"Correlation threshold increments: {unwrap.correlation_threshold}")
        info.log(f"Good correlation: {unwrap.good_correlation}")
        info.log(
            f"Minimum size of an unwrapped region: {unwrap.min_pixels_region}")

    return info


def set_icu_attributes(cfg: dict):
    """
    Return ICU object with user-defined attribute values

    Parameters
    ----------
    cfg: dict
        Dictionary containing user-defined ICU parameters

    Returns
    -------
    unwrap: isce3.unwrap.ICU
        ICU object with user-defined attribute values
    """
    unwrap = isce3.unwrap.ICU()
    unwrap.corr_incr_thr = cfg["correlation_threshold_increments"]
    unwrap.buffer_lines = cfg["buffer_lines"]
    unwrap.overlap_lines = cfg["overlap_lines"]
    unwrap.use_phase_grad_neut = cfg["use_phase_gradient_neutron"]
    unwrap.use_intensity_neut = cfg["use_intensity_neutron"]
    unwrap.phase_grad_win_size = cfg["phase_gradient_window_size"]
    unwrap.neut_phase_grad_thr = cfg["neutron_phase_gradient_threshold"]
    unwrap.neut_intensity_thr = cfg["neutron_intensity_threshold"]
    unwrap.neut_correlation_thr = cfg["max_intensity_correlation_threshold"]
    unwrap.trees_number = cfg["trees_number"]
    unwrap.max_branch_length = cfg["max_branch_length"]
    unwrap.ratio_dxdy = cfg["pixel_spacing_ratio"]
    unwrap.init_corr_thr = cfg["initial_correlation_threshold"]
    unwrap.max_corr_thr = cfg["max_correlation_threshold"]
    unwrap.min_cc_area = cfg["min_tile_area"]
    unwrap.num_bs_lines = cfg["bootstrap_lines"]
    unwrap.min_overlap_area = cfg["min_overlap_area"]
    unwrap.phase_var_thr = cfg["phase_variance_threshold"]

    return unwrap


def set_phass_attributes(cfg: dict):
    """
    Return Phass object with user-defined attribute values

    Parameters
    ----------
    cfg: dict
        Dictionary containing Phass parameters

    Returns
    -------
    unwrap: isce3.unwrap.Phass
        Phass object with user-defined attribute values
    """
    unwrap = isce3.unwrap.Phass()

    unwrap.correlation_threshold = cfg["correlation_threshold_increments"]
    unwrap.good_correlation = cfg["good_correlation"]
    unwrap.min_pixels_region = cfg["min_unwrap_area"]

    return unwrap


def igram_phase_to_vrt(raster_path, output_path):
    """
    Save the phase of complex raster in 'raster_path'
    in a GDAL VRT format

    Parameters
    ----------
    raster_path: str
        File path of complex raster to save in VRT format
    output_path: str
        File path of output phase VRT raster
    """

    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    vrttmpl = f"""
            <VRTDataset rasterXSize="{ds.RasterXSize}" rasterYSize="{ds.RasterYSize}">
            <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
            <Description>Phase</Description>
            <PixelFunctionType>phase</PixelFunctionType>
            <SimpleSource>
            <SourceFilename>{raster_path}</SourceFilename>
            </SimpleSource>
            </VRTRasterBand>
            </VRTDataset>"""
    ds = None
    with open(output_path, "w") as fid:
        fid.write(vrttmpl)


def get_effective_looks(rslc_obj, orbit_obj, rg_spac,
                        az_spac, rg_bw, az_bw, freq='A'):
    """
    Compute the effective number of looks based on RSLC
    range and azimuth spacings and bandwidths

    Parameters
    ----------
    rslc_obj: isce3.nisar.products.readers.SLC
        ISCE3 RSLC object
    orbit_obj: isce3.core.Orbit
        ISCE3 orbit object
    rg_spac: float
        Spacing (in meters) of the wrapped interferogram
        in the range direction
    az_spac: float
        Spacing (in meters) of the wrapped interferogram
        in the along-track direction
    rg_bw: float
        Range bandwidth (in Hz) of the wrapped interferogram
    az_bw: float
        Azimuth bandwidth (in Hz) of the wrapped interferogram
    freq: str
        NISAR frequency band code: 'A' for main band, 'B' for
        side band (default: 'A')

    Returns
    ------
    nlooks: float
        The equivalent number of independent looks used to form the sample coherence. An
        estimate of the number of statistically independent samples averaged in the
        multilooked data, taking into account spatial correlation due to
        oversampling/filtering (see `Notes`_)

    Notes
    -----
    An estimate of the equivalent number of independent looks may be obtained by

    .. math:: n_e = k_r k_a \frac{d_r d_a}{\rho_r \rho_a}

    where :math:`k_r` and :math:`k_a` are the number of looks in range and azimuth,
    :math:`d_r` and :math:`d_a` are the (single-look) sample spacing in range and
    azimuth, and :math:`\rho_r` and :math:`\rho_a are the range and azimuth resolution.
    """

    # Get radar grid
    radar_grid = rslc_obj.getRadarGrid(freq)

    # Compute range resolution
    rg_res = isce3.core.speed_of_light / (2.0 * rg_bw)
    _, v_mid = orbit_obj.interpolate(radar_grid.sensing_mid)
    vs = np.linalg.norm(v_mid)
    az_res = vs / az_bw
    nlooks = rg_spac * az_spac / (rg_res * az_res)

    return nlooks


def open_raster(filename, band=1):
    """
    Open GDAL-friendly raster and allocate 'band'
    in numpy array

    Parameters
    ----------
    filename: str
        Path to the GDAL-friendly raster
    band: int
        Band number to extract

    Returns
    -------
    raster: np.ndarray
        Raster band allocated in numpy array
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    raster = ds.GetRasterBand(band).ReadAsArray()
    return raster


def write_raster(filename, array, data_type=gdal.GDT_CFloat32,
                 file_format='ENVI'):
    '''
    Write numpy array to a GDAL-friendly file

    Parameters
    ----------
    filename: str
        Output file path for array to write to disk
    array: np.ndarray
        Numpy array to write to disk
    data_type: gdal.DataType
        GDAL data type (default: gdal.GDT_CFloat32)
    file_format: str
        GDAL file format (default: ENVI)
    '''

    driver = gdal.GetDriverByName(file_format)
    length, width = array.shape
    out_ds = driver.Create(filename, width, length, 1,
                           data_type)
    out_ds.GetRasterBand(1).WriteArray(array)
    out_ds.FlushCache()


if __name__ == "__main__":
    """
    Run phase unwrapping from command line
    """

    # Load command line args
    unwrap_parser = YamlArgparse()
    args = unwrap_parser.parse()

    # Get a runconfig dictionary from command line args
    unwrap_runconfig = UnwrapRunConfig(args)

    # Prepare RUNW HDF5
    unwrap_runconfig.cfg["primary_executable"]["product_type"] =\
        "RUNW_STANDALONE"
    out_paths = prepare_insar_hdf5.run(unwrap_runconfig.cfg)

    # Use RIFG from crossmul_path
    rifg_h5 = \
        unwrap_runconfig.cfg["processing"]["phase_unwrap"]["crossmul_path"]

    # Run phase unwrapping
    run(unwrap_runconfig.cfg, rifg_h5, out_paths["RUNW"])
