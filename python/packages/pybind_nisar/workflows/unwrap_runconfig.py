from collections import defaultdict
import os

import journal

from pybind_nisar.workflows.runconfig import RunConfig
import pybind_nisar.workflows.helpers as helpers


class UnwrapRunConfig(RunConfig):
    def __init__(self, args):
        super().__init__(args, 'insar')

        if self.args.run_config_path is None:
            self.cli_arg_load()
        else:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def cli_arg_load(self):

        error_channel = journal.error('UnwrapRunConfig.cli_arg_load')
        self.cfg = helpers.autovivified_dict

        # Valid crossmul path
        if os.path.isfile(self.args.crossmul):
            self.cfg['processing']['phase_unwrap']['crossmul_path'] = self.args.crossmul
        else:
            err_str = f"{self.args.crossmul} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        # Valid seed?
        if isinstance(self.args.seed, int):
            self.cfg['processing']['phase_unwrap']['seed'] = self.args.seed
        else:
            err_str = f"{self.args.seed} not a valid value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid number of buffer lines?
        if self.args.buffer_lines >= 1:
            self.cfg['processing']['phase_unwrap']['buffer_lines'] = self.args.buffer_lines
        else:
            err_str = f"{self.args.buffer_lines} needs to be greater than 0"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid number of overlapping lines?
        if self.args.overlap_lines >= 1:
            self.cfg['processing']['phase_unwrap']['overlap_lines'] = self.args.overlap_lines
        else:
            err_str = f"{self.args.overlap_lines} needs to be greater than 0"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid phase neutron flag?
        if isinstance(self.args.use_phase, bool):
            self.cfg['processing']['phase_unwrap']['use_phase_gradient_neutron'] = self.args.use_phase
        else:
            err_str = f"{self.args.use_phase} not a valid value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid intensity flag?
        if isinstance(self.args.use_intensity, bool):
            self.cfg['processing']['phase_unwrap']['use_intensity_neutron'] = self.args.use_intensity
        else:
            err_str = f"{self.args.use_intensity} not a valid value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid phase gradient window size?
        if self.args.pha_grad_win >=1 :
            self.cfg['processing']['phase_unwrap']['phase_gradient_window_size'] = self.args.pha_grad_win
        else:
            err_str = f"{self.args.pha_grad_win} needs to be greater than 0"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid neutron phase gradient threshold?
        if self.args.neut_pha_grad_thr >= 0:
            self.cfg['processing']['phase_unwrap']['neutron_phase_gradient_threshold'] = self.args.neut_pha_grad_thr
        else:
            err_str = f"{self.args.neut_pha_grad_thr} Needs to be greater or equal to zero"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid neutron intensity threshold?
        if self.args.neut_int_thr > 0.0:
            self.cfg['processing']['phase_unwrap']['neutron_intensity_threshold'] = self.args.neut_int_thr
        else:
            err_str = f"{self.args.neut_int_thr} needs to be a positive value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid max intensity correlation threshold?
        if 0 < self.args.neut_corr_thr < 1:
            self.cfg['processing']['phase_unwrap']['max_intensity_correlation_threshold'] = self.args.neut_corr_thr
        else:
            err_str = f"{self.args.neut_corr_thr} needs to be in [0, 1]"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid number of trees?
        if self.args.trees_number >= 1:
            self.cfg['processing']['phase_unwrap']['trees_number'] = self.args.trees_number
        else:
            err_str = f"{self.args.trees_number} needs to be greater than 0"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid max branch length?
        if self.args.max_branch_length >= 1:
            self.cfg['processing']['phase_unwrap']['max_branch_length'] = self.args.max_branch_length
        else:
            err_str = f"{self.args.max_branch_length} needs to be greater than 0"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid pixel spacing ratio?
        if self.args.ratio_dxdy > 0:
            self.cfg['processing']['phase_unwrap']['pixel_spacing_ratio'] = self.args.ratio_dxdy
        else:
            err_str = f"{self.args.ratio_dxdy} needs to be a positive value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid initial correlation threshold?
        if 0 < self.args.init_corr_thr < 1:
            self.cfg['processing']['phase_unwrap']['initial_correlation_threshold'] = self.args.init_corr_thr
        else:
            err_str = f"{self.args.init_corr_thr} needs to be in [0, 1]"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid max correlation threshold?
        if 0 < self.args.max_corr_thr < 1:
            self.cfg['processing']['phase_unwrap']['max_correlation_threshold'] = self.args.max_corr_thr
        else:
            err_str = f"{self.args.max_corr_thr} needs to be in [0, 1]"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid correlation threshold increments?
        if 0 < self.args.corr_incr_thr < 1:
            self.cfg['processing']['phase_unwrap']['correlation_threshold_increments'] = self.args.corr_incr_thr
        else:
            err_str = f"{self.args.corr_incr_thr} needs to be in [0, 1]"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid connected components area?
        if self.args.min_cc_area > 0:
            self.cfg['processing']['phase_unwrap']['min_tile_area'] = self.args.min_cc_area
        else:
            err_str = f"{self.args.min_cc_area} needs to be a positive value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid number of bootstrap lines?
        if self.args.num_bs_lines >= 1:
            self.cfg['processing']['phase_unwrap']['bootstrap_lines'] = self.args.num_bs_lines
        else:
            err_str = f"{self.args.num_bs_lines} needs to be greater than zero"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid value for min overlapping area
        if self.args.min_overlap_area >= 0:
            self.cfg['processing']['phase_unwrap']['min_overlap_area'] = self.args.min_overlap_area
        else:
            err_str = f"{self.args.min_overlap_area} needs to be a positive value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Valid phase variance threshold?
        if self.args.pha_var_thr > 0:
            self.cfg['processing']['phase_unwrap']['phase_variance_threshold'] = self.args.pha_var_thr
        else:
            err_str = f"{self.args.pha_var_thr} needs to be a positive value"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # check frequency and polarization dict prior to dict assignment
        freq_pols = self.args.freq_pols
        for k, vals in freq_pols.items():
            # check if frequency key is valid
            if k not in ['A', 'B']:
                err_str = f"frequency {k} not valid"
                error_channel.log(err_str)
                raise ValueError(err_str)
            # check if polarization values are valid
            for val in vals:
                if val not in ['HH', 'VV', 'HV', 'VH']:
                    err_str = f"polarization {val} not valid"
                    error_channel.log(err_str)
                    raise ValueError(err_str)

        self.cfg['processing']['input_subset']['list_of_frequencies'] = freq_pols

        # Check output directory
        outdir = os.path.dirname(self.args.output_h5)
        helpers.check_write_dir(outdir)
        self.cfg['ProductPathGroup']['SASOutputFile'] = self.args.output_h5

        # Not sure why this ...
        self.cfg['PrimaryExecutable']['ProductType'] = 'RIFG'

    def yaml_check(self):
        '''
        Check phase_unwrap specifics from YAML
        '''
        error_channel = journal.error('CrossmulRunConfig.yaml_check')

        if self.cfg['processing']['phase_unwrap'] is None:
            err_str = "'phase_unwrap' necessary for standalone execution with YAML"
            error_channel.log(err_str)
            raise ValueError(err_str)

        if 'crossmul_path' not in self.cfg['processing']['phase_unwrap']:
            err_str = "'crossmul_path' file path under `phase_unwrap' required for standalone execution with YAML"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if crossmul path is a directory or a file
        crossmul_path = self.cfg['processing']['phase_unwrap']['crossmul_path']
        if not os.path.isfile(crossmul_path):
            err_str = f"{crossmul_path} is invalid; needs to be a file"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if required polarizations/frequency are in crossmul_path file
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        helpers.check_hdf5_freq_pols(crossmul_path, freq_pols)
