import os

import journal
from nisar.workflows.helpers import check_mode_directory_tree, get_cfg_freq_pols
from nisar.workflows.runconfig import RunConfig


class ResampleSlcRunConfig(RunConfig):
    def __init__(self, args, resample_type='coarse'):
        # InSAR submodules have a common InSAR schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            super().load_geocode_yaml_to_dict()
            super().geocode_common_arg_load()
            self.yaml_check(resample_type)

    def yaml_check(self, resample_type):
        '''
        Check resample specifics from YAML.
        '''
        error_channel = journal.error('ResampleSlcRunConfig.yaml_check')

        # Extract frequency
        freq_pols = self.cfg['processing']['input_subset'][
            'list_of_frequencies']
        frequencies = freq_pols.keys()

        if resample_type not in ['coarse', 'fine']:
            err_str = f"{resample_type} is not a valid resample mode"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # For insar.py, offsets_dir comes from the previous step of the
        # workflow through scratch_path
        resample_key = f'{resample_type}_resample'
        if self.cfg['processing'][resample_key]['offsets_dir'] is None:
            self.cfg['processing'][resample_key]['offsets_dir'] = \
                self.cfg['product_path_group']['scratch_path']
        offsets_dir = self.cfg['processing'][resample_key]['offsets_dir']

        # Check directory structure and existence of offset files depending on
        # the selected resample type
        if resample_type == 'coarse':
            check_mode_directory_tree(offsets_dir, 'geo2rdr',
                                      frequencies)
            for freq in frequencies:
                rg_off = os.path.join(offsets_dir, 'geo2rdr', f'freq{freq}',
                                      'range.off')
                az_off = rg_off.replace('range', 'azimuth')
                if not os.path.exists(rg_off) or not os.path.exists(az_off):
                    err_str = f'{rg_off} and {az_off} offsets files do not exist'
                    error_channel.log(err_str)
                    raise FileNotFoundError(err_str)
        else:
            # use the HH or VV rubbersheeted offsets to fine
            # resample the secondary SLC. Check for the offsets existence
            for freq, _, pol_list in get_cfg_freq_pols(self.cfg):
                for pol in pol_list:
                    rg_off = os.path.join(offsets_dir,
                                          'rubbersheet_offsets',
                                          f'freq{freq}', pol, 'range.off')
                    az_off = rg_off.replace('range', 'azimuth')
                    if not os.path.exists(rg_off) or not os.path.exists(az_off):
                        err_str = f"{rg_off} and {az_off} files do not exists. HH and" \
                                  f"VV rubbersheet offsets required to run fine resampling"
                        error_channel.log(err_str)
                        raise FileNotFoundError(err_str)
