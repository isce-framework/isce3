#!/usr/bin/env python3
import time
import os
import journal
import pathlib

import numpy as np
from osgeo import gdal
import h5py
import copy

from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.ionosphere_runconfig import InsarIonosphereRunConfig
from nisar.workflows import (crossmul, dense_offsets, h5_prep,
                             filter_interferogram, resample_slc, 
                             rubbersheet, unwrap)
from isce3.splitspectrum import splitspectrum
from nisar.products.readers import SLC
import isce3
from nisar.workflows.filter_data import filter_data
from nisar.workflows.filter_interferogram import create_gaussian_kernel
from isce3.ionosphere import ionosphere_estimation


def write_disp_block_hdf5(
        hdf5_str, 
        path, 
        data,
        rows, 
        block_row=0):
    """write block array to HDF5
    
    Parameters
    ----------
    hdf5_str : str 
        output HDF5 file name 
    path : str
        HDF5 path for dataset
    data : numpy.ndarray
        block data to be saved to HDF5
    rows : int
        number of rows of entire data
    block_row : int
        block start index
    """    
    # If hdf5 file and path exists, then write the data into file. 
    try:
        with h5py.File(hdf5_str, 'r+') as dst_h5:
            block_length, block_width = data.shape
            dst_h5[path].write_direct(data,
                dest_sel=np.s_[block_row : block_row + block_length, 
                    :block_width])
    except:
        pass

def insar_ionosphere_pair(cfg):
    """Run insar workflow for ionosphere pairs
    
    Parameters
    ----------
    cfg : dict
        dictionary of runconfigs
    """     
    
    # ionosphere runconfigs
    iono_args = cfg['processing']['ionosphere_phase_correction']
    scratch_path = cfg['product_path_group']['scratch_path']
  
    # pull parameters for ionosphere phase estimation
    iono_freq_pols = iono_args['list_of_frequencies']
    iono_method = iono_args['spectral_diversity']

    iono_path = os.path.join(scratch_path, 'ionosphere')
    split_slc_path = os.path.join(iono_path, 'split_spectrum')

    # Keep cfg before changing it
    orig_scratch_path = scratch_path  
    orig_ref_str = cfg['input_file_group']['reference_rslc_file_path']
    orig_sec_str = cfg['input_file_group']['secondary_rslc_file_path']
    orig_coreg_path = cfg['processing']['crossmul'][
        'coregistered_slc_path']
    orig_freq_pols = copy.deepcopy(cfg['processing']['input_subset'][
                    'list_of_frequencies'])
    orig_product_type = cfg['primary_executable']['product_type']
    iono_insar_cfg = cfg.copy()

    iono_insar_cfg['primary_executable'][
                'product_type'] = 'RUNW'
    if iono_method == 'split_main_band':
        for split_str in ['low', 'high']:

            # update reference sub-band path 
            ref_h5_path = os.path.join(split_slc_path, 
                f"ref_{split_str}_band_slc.h5")
            iono_insar_cfg['input_file_group'][
                'reference_rslc_file_path'] = ref_h5_path

            # update secondary sub-band path 
            sec_h5_path = os.path.join(split_slc_path, 
                f"sec_{split_str}_band_slc.h5")                
            iono_insar_cfg['input_file_group'][
                'secondary_rslc_file_path'] = sec_h5_path
            
            # update output path 
            new_scratch = pathlib.Path(orig_scratch_path, 
                'ionosphere', split_str)                
            iono_insar_cfg['product_path_group'][
                'scratch_path'] = new_scratch
            iono_insar_cfg['product_path_group'][
                'sas_output_file'] = f'{new_scratch}/RUNW.h5'
            iono_insar_cfg['processing']['dense_offsets'][
                'coregistered_slc_path'] = new_scratch
            iono_insar_cfg['processing']['crossmul'][
                'coregistered_slc_path'] = new_scratch

            # update frequency and polarizations for ionosphere
            if iono_freq_pols['A']:
                iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies']['A'] = iono_freq_pols['A']
            if iono_freq_pols['B']:
                iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies']['B'] = iono_freq_pols['B']
            else:
                # if cfg has key for frequency B, then delete it to avoid 
                # unnecessary insar processing
                try:
                    del iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies']['B']
                except:
                    pass

            # create directory for sub-band interferograms
            new_scratch.mkdir(parents=True, exist_ok=True)

            # run insar for sub-band SLCs
            _, out_paths = h5_prep.get_products_and_paths(iono_insar_cfg)
            out_paths['RUNW'] = f'{new_scratch}/RUNW.h5'
            run_insar_workflow(iono_insar_cfg, out_paths)
          
    elif iono_method in ['main_side_band', 'main_diff_ms_band']:
        rerun_insar_pairs = 0
        for freq in iono_freq_pols.keys():
            iono_pol = iono_freq_pols[freq]
            try:
                orig_pol = orig_freq_pols[freq]
            except:
                orig_pol = []
            res_pol = [pol for pol in iono_pol if pol not in orig_pol]    

            # update frequency and polarizations for ionosphere
            if res_pol:
                iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies'][freq] = res_pol
                rerun_insar_pairs =+ 1
            else: 
                del iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies'][freq] 
                   
        if rerun_insar_pairs > 0 :
            # update paths 
            new_scratch = pathlib.Path(iono_path, f'{iono_method}')
            iono_insar_cfg['product_path_group'][
                'scratch_path'] = new_scratch
            iono_insar_cfg['product_path_group'][
                    'sas_output_file'] = f'{new_scratch}/RUNW.h5'
            iono_insar_cfg['processing']['dense_offsets'][
                'coregistered_slc_path'] = new_scratch
            iono_insar_cfg['processing']['crossmul'][
                'coregistered_slc_path'] = new_scratch

            new_scratch.mkdir(parents=True, exist_ok=True)

            _, out_paths = h5_prep.get_products_and_paths(iono_insar_cfg)
            out_paths['RUNW'] = f'{new_scratch}/RUNW.h5'
            run_insar_workflow(iono_insar_cfg, out_paths)
        
    # restore original paths
    cfg['input_file_group']['reference_rslc_file_path'] = orig_ref_str
    cfg['input_file_group']['secondary_rslc_file_path'] = orig_sec_str
    cfg['product_path_group']['scratch_path'] = orig_scratch_path
    cfg['processing']['dense_offsets'][
        'coregistered_slc_path'] = orig_coreg_path
    cfg['processing']['crossmul'][
        'coregistered_slc_path'] = orig_coreg_path
    cfg['processing']['input_subset'][
            'list_of_frequencies'] = orig_freq_pols
    cfg['primary_executable'][
                'product_type'] = orig_product_type
    cfg['processing']['geo2rdr']['topo_path'] = orig_scratch_path

def run_insar_workflow(cfg, out_paths):
    # run insar for ionosphere pairs
    h5_prep.run(cfg)

    resample_slc.run(cfg, 'coarse')

    if cfg['processing']['fine_resample']['enabled']:
        resample_slc.run(cfg, 'fine')

    if cfg['processing']['fine_resample']['enabled']:
        crossmul.run(cfg, out_paths['RIFG'], 'fine')
    else:
        crossmul.run(cfg, out_paths['RIFG'], 'coarse')

    if cfg['processing']['filter_interferogram']['filter_type'] != 'no_filter':
        filter_interferogram.run(cfg, out_paths['RIFG'])

    if 'RUNW' in out_paths:
        unwrap.run(cfg, out_paths['RIFG'], out_paths['RUNW'])

def run(cfg: dict, runw_hdf5: str):
    '''
    Run ionosphere phase correction workflow with parameters 
    in cfg dictionary
    Parameters
    ---------
    cfg: dict
        Dictionary with user-defined options
    runw_hdf5: str
        File path to runw HDF5 product (i.e., RUNW)
    '''
    
    # Create error and info channels
    info_channel = journal.info("ionosphere_phase_correction.run")
    info_channel.log("starting insar_ionosphere_correction")

    # pull parameters from dictionary
    iono_args = cfg['processing']['ionosphere_phase_correction']
    scratch_path = cfg['product_path_group']['scratch_path']

    # pull parameters for ionosphere phase estimation
    iono_freq_pols = copy.deepcopy(iono_args['list_of_frequencies'])
    iono_method = iono_args['spectral_diversity']
    blocksize = iono_args['lines_per_block']
    filter_cfg = iono_args['dispersive_filter']

    # pull parameters for dispersive filter
    filter_bool =filter_cfg['enabled']
    mask_type = filter_cfg['filter_mask_type']
    filter_coh_thresh = filter_cfg['filter_coherence_threshold']
    kernel_x_size = filter_cfg['kernel_x']
    kernel_y_size = filter_cfg['kernel_y']
    kernel_sigma_x = filter_cfg['sigma_x']
    kernel_sigma_y = filter_cfg['sigma_y']
    filling_method = filter_cfg['filling_method']
    filter_iterations = filter_cfg['filter_iterations']
    unwrap_correction_bool = filter_cfg['unwrap_correction']
    rg_looks = cfg['processing']['crossmul']['range_looks']
    az_looks = cfg['processing']['crossmul']['azimuth_looks']

    t_all = time.time()

    # set paths for ionosphere and split spectrum
    iono_path = os.path.join(scratch_path, 'ionosphere')
    split_slc_path = os.path.join(iono_path, 'split_spectrum')

    # Keep cfg before changing it
    orig_scratch_path = cfg['product_path_group']['scratch_path']   
    orig_ref_str = cfg['input_file_group']['reference_rslc_file_path']
    orig_sec_str = cfg['input_file_group']['secondary_rslc_file_path']
    orig_freq_pols = copy.deepcopy(cfg['processing']['input_subset'][
                    'list_of_frequencies'])
    iono_insar_cfg = cfg.copy()

    # Run InSAR for sub-band SLCs (split-main-bands) or 
    # for main and side bands for iono_freq_pols (main-side-bands)
    insar_ionosphere_pair(iono_insar_cfg)
              
    # Define methods to use subband or sideband
    iono_method_subbands = ['split_main_band']
    iono_method_sideband = ['main_side_band', 'main_diff_ms_band']

    # set frequency A RUNW path 
    if runw_hdf5:
        runw_path_insar = runw_hdf5
    else:
        runw_path_insar = os.path.join(scratch_path, 'RUNW.h5')

    # Start ionosphere phase estimation 
    # pull center frequency from frequency A, which is used for all method
    base_ref_slc_str = orig_ref_str
    base_ref_slc = SLC(hdf5file=base_ref_slc_str)
    ref_meta_data_a = splitspectrum.bandpass_meta_data.load_from_slc(
        slc_product=base_ref_slc, 
        freq='A')
    f0 = ref_meta_data_a.center_freq

    if iono_method in iono_method_subbands:
        # pull center frequencies from sub-bands
        high_ref_slc_str = os.path.join(split_slc_path, f"ref_high_band_slc.h5")
        low_ref_slc_str = os.path.join(split_slc_path, f"ref_low_band_slc.h5")
        high_ref_slc = SLC(hdf5file=high_ref_slc_str)
        low_ref_slc = SLC(hdf5file=low_ref_slc_str)

        high_sub_meta_data = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=high_ref_slc, 
            freq='A')
        low_sub_meta_data = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=low_ref_slc, 
            freq='A')
        f0_low = low_sub_meta_data.center_freq
        f0_high = high_sub_meta_data.center_freq

    if iono_method in iono_method_sideband:
        # pull center frequency from frequency B
        ref_meta_data_b = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=base_ref_slc, 
            freq='B')                
        f1 = ref_meta_data_b.center_freq
        
        # find polarizations which are not processed in InSAR workflow 
        residual_pol_a =  list(set(
            iono_freq_pols['A']) - set(orig_freq_pols['A'])) 
        residual_pol_b =  list(set(
            iono_freq_pols['B']) - set(orig_freq_pols['B']))  

    if iono_method == 'split_main_band':
        f1 = None
    elif iono_method in ['main_side_band', 'main_diff_ms_band']:
        f0_low = None
        f0_high = None        

    # Create object for ionosphere esimation
    iono_phase_obj = ionosphere_estimation.IonosphereEstimation(
        main_center_freq=f0,
        side_center_freq=f1, 
        low_center_freq=f0_low, 
        high_center_freq=f0_high,
        method=iono_method)

    # Create object for ionosphere filter
    iono_filter_obj = ionosphere_estimation.IonosphereFilter(
        x_kernel=kernel_x_size, 
        y_kernel=kernel_y_size, 
        sig_x=kernel_sigma_x, 
        sig_y=kernel_sigma_y, 
        iteration=filter_iterations,
        filling_method=filling_method,
        outputdir=os.path.join(iono_path, iono_method))

    # pull parameters for polarizations 
    pol_list_a = iono_freq_pols['A']
    pol_list_b = iono_freq_pols['B']
    # Read Block and estimate dispersive and non-dispersive
    for pol_ind, pol_a in enumerate(pol_list_a):
   
        # pull array for sub-bands
        pol_comb_str = f"{pol_a}_{pol_a}"     
        swath_path = f"/science/LSAR/RUNW/swaths"
        dest_freq_path = f"{swath_path}/frequencyA"
        dest_pol_path = f"{dest_freq_path}/interferogram/{pol_a}"
        runw_path_freq_a = f"{dest_pol_path}/unwrappedPhase"
        rcoh_path_freq_a = f"{dest_pol_path}/coherenceMagnitude"
        rcom_path_freq_a = f"{dest_pol_path}/connectedComponents"
        rslant_path_a = f"{dest_freq_path}/interferogram/"\
            "slantRange"

        if iono_method in iono_method_sideband:
            pol_b = pol_list_b[pol_ind]
            pol_comb_str = f"{pol_a}_{pol_b}"     
            dest_freq_path_b = f"{swath_path}/frequencyB"
            dest_pol_path_b = f"{dest_freq_path_b}/interferogram/{pol_b}"
            runw_path_freq_b = f"{dest_pol_path_b}/unwrappedPhase"
            rcoh_path_freq_b = f"{dest_pol_path_b}/coherenceMagnitude"
            rcom_path_freq_b = f"{dest_pol_path_b}/connectedComponents"
            rslant_path_b = f"{dest_freq_path_b}/interferogram/"\
                "slantRange"
            
        if iono_method in iono_method_subbands:
            # set paths for high and low sub-bands
            sub_low_runw_str = os.path.join(iono_path, 'low', 'RUNW.h5')
            sub_high_runw_str = os.path.join(iono_path, 'high', 'RUNW.h5')

            target_array_str = f'HDF5:{sub_low_runw_str}:/{runw_path_freq_a}'
            target_slc_array = isce3.io.Raster(target_array_str)   
            rows_main = target_slc_array.length   
            cols_main = target_slc_array.width         
            nblocks = int(np.ceil(rows_main / blocksize))
            rows_output = rows_main
            cols_output = cols_main
            # In method using only sub-bands, resampling is unnecessary. 
            # thus, slant range info is not needed. 
            main_slant = None
            side_slant = None

        if iono_method in iono_method_sideband:
            # set paths for frequency A 
            if pol_a in residual_pol_a:
                runw_freq_a_str = os.path.join(
                    iono_path, iono_method, 'RUNW.h5')
            # If target polarization is in pre-existing HDF5, 
            # then use it without additional InSAR workflow. 
            else:
                # runw_freq_a_str = os.path.join(
                #     scratch_path, 'RUNW.h5')
                runw_freq_a_str = runw_path_insar
            # set paths for frequency B 
            if pol_b in residual_pol_b:
                runw_freq_b_str = os.path.join(iono_path, 
                    iono_method, 'RUNW.h5')
            else:
                runw_freq_b_str = runw_path_insar 
                
            main_array_str = f'HDF5:{runw_freq_a_str}:/{runw_path_freq_a}'
            main_runw_array = isce3.io.Raster(main_array_str)   
            rows_main = main_runw_array.length   
            cols_main = main_runw_array.width         
            nblocks = int(np.ceil(rows_main / blocksize))

            side_array_str = f'HDF5:{runw_freq_b_str}:/{runw_path_freq_b}'
            side_runw_array = isce3.io.Raster(side_array_str)   
            rows_side = side_runw_array.length   
            cols_side = side_runw_array.width         

            main_slant = np.empty([cols_main], dtype=float)
            side_slant = np.empty([cols_side], dtype=float)
            rows_output = rows_side
            cols_output = cols_side
            # Read slant range array 
            with h5py.File(runw_freq_a_str, 'r', 
                libver='latest', swmr=True) as src_main_h5, \
                h5py.File(runw_freq_b_str, 'r',
                libver='latest', swmr=True) as src_side_h5:
                
                # Read slant range block from HDF5
                src_main_h5[rslant_path_a].read_direct(
                    main_slant, np.s_[:])
                src_side_h5[rslant_path_b].read_direct(
                    side_slant, np.s_[:])
            
        for block in range(0, nblocks):
            info_channel.log("Ionosphere Phase Estimation block: ", block)
            
            row_start = block * blocksize
            if (row_start + blocksize > rows_main):
                block_rows_data = rows_main - row_start
            else:
                block_rows_data = blocksize
            
            # initialize arrays by setting None
            sub_low_image = None
            sub_high_image = None
            main_image = None
            side_image = None
            
            sub_low_coh_image = None
            sub_high_coh_image = None
            main_coh_image = None
            side_coh_image = None

            sub_low_conn_image = None
            sub_high_conn_image = None
            main_conn_image = None
            side_conn_image = None

            if iono_method in iono_method_subbands:
                sub_low_image = np.empty([block_rows_data, cols_main], 
                    dtype=float)
                sub_high_image = np.empty([block_rows_data, cols_main], 
                    dtype=float)
                sub_low_coh_image = np.empty([block_rows_data, cols_main], 
                    dtype=float)    
                sub_high_coh_image = np.empty([block_rows_data, cols_main], 
                    dtype=float)
                
                if mask_type == "connected_components":
                    sub_low_conn_image = np.empty(
                        [block_rows_data, cols_main], 
                        dtype=float)    
                    sub_high_conn_image = np.empty(
                        [block_rows_data, cols_main], 
                        dtype=float)

                with h5py.File(sub_low_runw_str, 'r', 
                    libver='latest', swmr=True) as src_low_h5, \
                    h5py.File(sub_high_runw_str, 'r',
                    libver='latest', swmr=True) as src_high_h5:
                    
                    # Read runw block for sub-bands
                    src_low_h5[runw_path_freq_a].read_direct(
                        sub_low_image, 
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_high_h5[runw_path_freq_a].read_direct(
                        sub_high_image, 
                        np.s_[row_start : row_start + block_rows_data, :])
                    # Read coherence block for sub-bands
                    src_low_h5[rcoh_path_freq_a].read_direct(
                        sub_low_coh_image, 
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_high_h5[rcoh_path_freq_a].read_direct(
                        sub_high_coh_image, 
                        np.s_[row_start : row_start + block_rows_data, :])

                    if mask_type == "connected_components":
                        # Read connected_components block for sub-bands
                        src_low_h5[rcom_path_freq_a].read_direct(
                            sub_low_conn_image, 
                            np.s_[row_start : row_start + block_rows_data, :])
                        src_high_h5[rcom_path_freq_a].read_direct(
                            sub_high_conn_image, 
                            np.s_[row_start : row_start + block_rows_data, :])

            if iono_method in iono_method_sideband:

                main_image = np.empty([block_rows_data, cols_main], 
                    dtype=float)
                side_image = np.empty([block_rows_data, cols_side], 
                    dtype=float)
                main_coh_image = np.empty([block_rows_data, cols_main], 
                    dtype=float)    
                side_coh_image = np.empty([block_rows_data, cols_side], 
                    dtype=float)

                if mask_type == "connected_components":
                    main_conn_image = np.empty([block_rows_data, cols_main], 
                        dtype=float)    
                    side_conn_image = np.empty([block_rows_data, cols_side], 
                        dtype=float)

                with h5py.File(runw_freq_a_str, 'r', 
                    libver='latest', swmr=True) as src_main_h5, \
                    h5py.File(runw_freq_b_str, 'r',
                    libver='latest', swmr=True) as src_side_h5:
                    
                    # Read runw block for main and side bands
                    src_main_h5[runw_path_freq_a].read_direct(
                        main_image, 
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_side_h5[runw_path_freq_b].read_direct(
                        side_image, 
                        np.s_[row_start : row_start + block_rows_data, :])
                    # Read coherence block for main and side bands
                    src_main_h5[rcoh_path_freq_a].read_direct(
                        main_coh_image, 
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_side_h5[rcoh_path_freq_b].read_direct(
                        side_coh_image, 
                        np.s_[row_start : row_start + block_rows_data, :])

                    if mask_type == "connected_components":
                        # Read connected components block for main and side bands
                        src_main_h5[rcom_path_freq_a].read_direct(
                            main_conn_image, 
                            np.s_[row_start : row_start + block_rows_data, :])
                        src_side_h5[rcom_path_freq_b].read_direct(
                            side_conn_image, 
                            np.s_[row_start : row_start + block_rows_data, :])                        
            
            # Estimate dispersive and non-dispersive phase
            dispersive, non_dispersive = iono_phase_obj.compute_disp_nondisp(
                phi_sub_low=sub_low_image, 
                phi_sub_high=sub_high_image,
                phi_main=main_image, 
                phi_side=side_image,
                slant_main=main_slant,
                slant_side=side_slant)

            # Write dispersive and non-dispersive phase into the 
            # ENVI format files            
            iono_method_path = pathlib.Path(iono_path, iono_method)
            iono_method_path.mkdir(parents=True, exist_ok=True)
            iono_pol_path = pathlib.Path(iono_method_path, pol_comb_str)
            iono_pol_path.mkdir(parents=True, exist_ok=True)
            
            out_disp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, f'dispersive')
            out_nondisp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'non_dispersive')
           
            ionosphere_estimation.write_array(out_disp_path, 
                dispersive, 
                data_type=gdal.GDT_Float32, 
                block_row=row_start, 
                data_shape=[rows_output, cols_output])
            ionosphere_estimation.write_array(out_nondisp_path, 
                non_dispersive, 
                data_type=gdal.GDT_Float32, 
                block_row=row_start, 
                data_shape=[rows_output, cols_output])

            # Calculating the theoretical standard deviation of the 
            # estimation based on the coherence of the interferograms   
            sig_phi_iono_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'dispersive.sig')
            sig_phi_nondisp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'nondispersive.sig')

            number_looks = rg_looks * az_looks

            iono_std, nondisp_std = iono_phase_obj.estimate_iono_std(
                main_coh=main_coh_image, 
                side_coh=side_coh_image,
                low_band_coh=sub_low_coh_image, 
                high_band_coh=sub_high_coh_image, 
                slant_main=main_slant,
                slant_side=side_slant,  
                number_looks=number_looks)

            # Write sigma of dispersive phase into the 
            # ENVI format files 
            ionosphere_estimation.write_array(sig_phi_iono_path, 
                iono_std, 
                data_type=gdal.GDT_Float32, 
                block_row=row_start, 
                data_shape=[rows_output, cols_output])
            ionosphere_estimation.write_array(sig_phi_nondisp_path, 
                nondisp_std, 
                data_type=gdal.GDT_Float32, 
                block_row=row_start, 
                data_shape=[rows_output, cols_output])
            # If filtering is not required, then write ionosphere phase 
            # at this point. 
            if not filter_bool:
                iono_hdf5_path = f'{dest_pol_path}/ionospherePhaseScreen'
                write_disp_block_hdf5(runw_path_insar, 
                    iono_hdf5_path, 
                    dispersive,
                    rows_output, 
                    row_start)

                iono_sig_hdf5_path = \
                    f'{dest_pol_path}/ionospherePhaseScreenUncertainty'
                write_disp_block_hdf5(runw_path_insar, 
                    iono_sig_hdf5_path, 
                    iono_std, 
                    rows_output,
                    row_start)  
            else:
                info_channel.log(f'{mask_type} is used for mask construction')

                if mask_type == "coherence":
                    mask_array = iono_phase_obj.get_mask_array( 
                        main_array=main_coh_image,
                        side_array=side_coh_image, 
                        low_band_array=sub_low_coh_image, 
                        high_band_array=sub_high_coh_image,
                        slant_main=main_slant,
                        slant_side=side_slant,  
                        threshold=filter_coh_thresh)

                elif mask_type == "connected_components":
                    mask_array = iono_phase_obj.get_mask_array( 
                        main_array=main_conn_image,
                        side_array=side_conn_image, 
                        low_band_array=sub_low_conn_image, 
                        high_band_array=sub_high_conn_image,
                        slant_main=main_slant,
                        slant_side=side_slant,  
                        threshold=0)

                elif mask_type == "median_filter":
                    mask_array = iono_phase_obj.get_mask_median_filter(
                        disp=dispersive,
                        looks=number_looks,
                        threshold=filter_coh_thresh)
                            
                mask_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'mask_array')
                # Write sigma of dispersive phase into the 
                # ENVI format files 
                ionosphere_estimation.write_array(mask_path, 
                    mask_array, 
                    data_type=gdal.GDT_Float32, 
                    block_row=row_start, 
                    data_shape=[rows_output, cols_output])

        # apply filter to entire scene to avoid discontinuity along 
        # block boundaries
        if filter_bool:
            disp_tif = gdal.Open(out_disp_path)
            dispersive = disp_tif.ReadAsArray()
            nondisp_tif = gdal.Open(out_nondisp_path)
            non_dispersive = nondisp_tif.ReadAsArray()
            mask_tif = gdal.Open(mask_path)
            mask_array = mask_tif.ReadAsArray()
            sig_disp_tif = gdal.Open(sig_phi_iono_path)
            iono_std = sig_disp_tif.ReadAsArray()
            sig_nondisp_tif = gdal.Open(sig_phi_nondisp_path)
            nondisp_std = sig_nondisp_tif.ReadAsArray()
            
            # low pass filtering for dispersive phase
            filt_disp, filt_data_sig = iono_filter_obj.low_pass_filter(
                input_array=dispersive, 
                input_sig=iono_std, 
                mask=mask_array)

            out_disp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'filt_dispersive')
            ionosphere_estimation.write_array(out_disp_path, 
                filt_disp, 
                data_type=gdal.GDT_Float32)                

            # low pass filtering for non-dispersive phase
            filt_nondisp, filt_nondisp_sig = iono_filter_obj.low_pass_filter(
                input_array=non_dispersive, 
                input_sig=nondisp_std, 
                mask=mask_array)

            out_nondisp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'filt_nondispersive')
            ionosphere_estimation.write_array(out_nondisp_path, 
                filt_nondisp, 
                data_type=gdal.GDT_Float32)

            # if unwrapping correction technique is not requested, 
            # save output to hdf5 at this point
            if not unwrap_correction_bool:
                iono_hdf5_path = f'{dest_pol_path}/ionospherePhaseScreen'
                write_disp_block_hdf5(runw_path_insar, 
                    iono_hdf5_path, 
                    filt_disp, 
                    rows_output)

                iono_sig_hdf5_path = \
                    f'{dest_pol_path}/ionospherePhaseScreenUncertainty'
                write_disp_block_hdf5(runw_path_insar, 
                    iono_sig_hdf5_path, 
                    filt_data_sig, 
                    rows_output)  
            else:
                # Estimating phase unwrapping errors
                com_unw_err, diff_unw_err = iono_phase_obj.compute_unwrapp_error(
                    disp_array=filt_disp, 
                    nondisp_array=filt_nondisp,
                    main_runw=main_image, 
                    side_runw=side_image,
                    low_sub_runw=sub_low_image, 
                    high_sub_runw=sub_high_image, 
                    y_ref=None, 
                    x_ref=None)

                dispersive_unwcor, non_dispersive_unwcor = \
                    iono_phase_obj.compute_disp_nondisp(
                    phi_sub_low=sub_low_image, 
                    phi_sub_high=sub_high_image,
                    phi_main=main_image, 
                    phi_side=side_image,
                    slant_main=main_slant,
                    slant_side=side_slant,
                    comm_unwcor_coef=com_unw_err,
                    diff_unwcor_coef=diff_unw_err)  

                filt_disp, filt_data_sig = iono_filter_obj.low_pass_filter(
                    input_array=dispersive_unwcor, 
                    input_sig=iono_std, 
                    mask=mask_array)

                out_disp_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'filt_dispersive')
                write_array(out_disp_path, 
                    filt_disp, 
                    data_type=gdal.GDT_Float32)   
                
                iono_hdf5_path = f'{dest_pol_path}/ionospherePhaseScreen'
                write_disp_block_hdf5(runw_path_insar, 
                    iono_hdf5_path, 
                    filt_disp, 
                    rows_output)

                iono_sig_hdf5_path = \
                    f'{dest_pol_path}/ionospherePhaseScreenUncertainty'
                write_disp_block_hdf5(runw_path_insar, 
                    iono_sig_hdf5_path, 
                    filt_data_sig, 
                    rows_output)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran INSAR in {t_all_elapsed:.3f} seconds")
                
if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    iono_runcfg = InsarIonosphereRunConfig(args)
    out_paths = h5_prep.run(iono_runcfg.cfg)

    run(iono_runcfg.cfg, runw_hdf5=out_paths['RUNW'])