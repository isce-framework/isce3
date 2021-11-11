import journal
from nisar.workflows.runconfig import RunConfig
from nisar.products.readers import SLC


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

        error_channel = journal.error('SplitSpectrumRunConfig.yaml_check')
        iono_cfg = self.cfg['processing']['ionosphere_phase_correction']

        # Extract main range bandwidth from reference RSLC
        ref_slc = SLC(hdf5file=self.cfg['InputFileGroup']['InputFilePath'])
        rg_main_bandwidth = ref_slc.getSwathMetadata('A').processed_range_bandwidth

        # Check if ionosphere_phase_correction is enabled. Otherwise,
        # throw an error and do not execute split-spectrum
        if not iono_cfg['enabled']:
            err_str = 'Ionosphere phase correction must be enabled to execute split-spectrum'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Depending on how the user has selected "spectral_diversity" check if
        # "low_bandwidth" and "high_bandwidth" are assigned. Otherwise, use default
        if iono_cfg['spectral_diversity'] == 'split_main_band':
            # If "low_bandwidth" or 'high_bandwidth" is not allocated, split the main range bandwidth
            # into two 1/3 sub-bands.
            if iono_cfg['low_bandwidth'] is None:
                iono_cfg['low_bandwidth'] = rg_main_bandwidth / 3.0
            if iono_cfg['high_bandwidth'] is None:
                iono_cfg['high_bandwidth'] = rg_main_bandwidth / 3.0

        if iono_cfg['spectral_diversity'] == 'main_side_band':
            # Extract side-band range bandwidth
            rg_side_bandwidth = ref_slc.getSwathMetadata('B').processed_range_bandwidth

            # If "low_bandwidth" and "high_bandwidth" are not assigned, assign main range bandwidth
            # and side-band bandwidths, respectively. If assigned, check that
            # "low_bandwidth" and "high_bandwidth" correspond to main and side range bandwidths
            if iono_cfg['low_bandwidth'] is None or iono_cfg['low_bandwidth'] != rg_main_bandwidth:
                iono_cfg['low_bandwidth'] = rg_main_bandwidth
            if iono_cfg['high_bandwidth'] is None or iono_cfg['high'] != rg_side_bandwidth:
                iono_cfg['high_bandwidth'] = rg_side_bandwidth

            # Check that main and side-band are at the same polarization. If not, throw an error.
            # Note, ref_slc.getSwathMetadata(freq) does not return list of polarizations.
            # TO DO: Modify the Swath class to return listOfPolarizations for freqA and freqB
