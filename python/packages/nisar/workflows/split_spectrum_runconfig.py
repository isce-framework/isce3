import journal

from nisar.workflows.runconfig import RunConfig
from nisar.workflows.ionosphere_runconfig import split_main_band_cfg_check


class SplitSpectrumRunConfig(RunConfig):
    def __init__(self, args):
        # All InSAR submodules share a common InSAR schema
        super().__init__(args, 'insar')

        # When using split spectrum as stand-alone module
        # check that the runconfig has been properly checked
        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check split-spectrum specifics from YAML file
        '''

        info_channel = journal.info('SplitSpectrumRunConfig.yaml_check')
        error_channel = journal.error('SplitSpectrumRunConfig.yaml_check')

        # Extract ionosphere options
        iono_cfg = self.cfg['processing']['ionosphere_phase_correction']
        iono_method = iono_cfg['spectral_diversity']

        if not iono_cfg['enabled']:
            err_str = f'Ionosphere phase correction must be enabled '\
                    f'to execute {iono_method}.'
            error_channel.log(err_str)
            raise ValueError(err_str)

        if not iono_cfg['enabled']:
            err_str = f'Ionosphere phase correction must be enabled '\
                    f'to execute {iono_method}.'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # check runconfigs using split_main_band in ionosphere
        if iono_method == 'split_main_band':
            split_main_band_cfg_check(self.cfg)
        else:
            err_str = f'Split_spectrum is not needed '\
                      f'to execute {iono_method}.'
            error_channel.log(err_str)
            raise ValueError(err_str)