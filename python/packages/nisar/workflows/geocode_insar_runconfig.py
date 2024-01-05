import os

import journal
import nisar.workflows.helpers as helpers
from nisar.workflows.runconfig import RunConfig


def geocode_insar_cfg_check(cfg):
    '''
    Check the geocode_insar runconfig configuration options

    Parameters
     ----------
     cfg: dict
        configuration dictionary

     Returns
     -------
     None
    '''
    if 'interp_method' not in cfg['processing']['geocode']:
        cfg['processing']['geocode']['interp_method'] = 'BILINEAR'

    # create empty dict if geocode_datasets not in geocode
    for datasets in ['gunw_datasets', 'goff_datasets', 'wrapped_datasets']:
        if datasets not in cfg['processing']['geocode']:
            cfg['processing']['geocode'][datasets] = {}

    # Initialize GUNW and GOFF names
    gunw_datasets = ['connected_components', 'coherence_magnitude',
                     'ionosphere_phase_screen',
                     'ionosphere_phase_screen_uncertainty',
                     'unwrapped_phase', 'along_track_offset',
                     'slant_range_offset', 'correlation_surface_peak',
                     'mask']
    goff_datasets = ['along_track_offset', 'snr',
                     'along_track_offset_variance',
                     'correlation_surface_peak', 'cross_offset_variance',
                     'slant_range_offset', 'slant_range_offset_variance']
    wrapped_datasets = ['coherence_magnitude', 'wrapped_interferogram']

    # insert both geocode datasets in dict keyed on datasets name
    geocode_datasets = {'gunw_datasets': gunw_datasets,
                        'goff_datasets': goff_datasets,
                        'wrapped_datasets': wrapped_datasets}
    for dataset_group in geocode_datasets:
        for dataset in geocode_datasets[dataset_group]:
            if dataset not in cfg['processing']['geocode'][dataset_group]:
                cfg['processing']['geocode'][dataset_group][dataset] = True

class GeocodeInsarRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a commmon `insar` schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check GUNW specifics from YAML
        '''
        error_channel = journal.error('GeocodeInsarRunConfig.yaml_check')

        if 'runw_path' not in self.cfg['processing']['geocode']:
            err_str = "'runw_path' file path under `geocode' required for standalone execution with YAML"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if runw path is a directory or a file
        runw_path = self.cfg['processing']['geocode']['runw_path']
        if not os.path.isfile(runw_path):
            err_str = f"{runw_path} is invalid; needs to be a file"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check if required polarizations/frequency are in provided HDF5 file
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        helpers.check_hdf5_freq_pols(runw_path, freq_pols)

        # Check geocode_insar runconfig values
        geocode_insar_cfg_check(self.cfg)
