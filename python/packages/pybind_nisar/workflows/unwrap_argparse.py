'''
class for parsing unwrapping args from CLI
'''

import argparse

from pybind_nisar.workflows.yaml_argparse import StoreDictKeyPair, YamlArgparse


class UnwrapArgparse(YamlArgparse):
    '''
    Class for parsing phase unwrapping args from CLI
    inherits YAML parsing from YamlArgparse
    '''

    def __init__(self):
        super().__init__()

        self.parser.add_argument('--crossmul', dest='crossmul', type=str,
                                 help='Path to the HDF5 containing interferogram and correlation')
        self.parser.add_argument('--output-h5', dest='output_h5', type=str,
                                 help="Path to the output dir")
        self.parser.add_argument('--seed', dest="seed", type=int, default=0,
                                 help="Seed value to initialize the ICU alfgorithm")
        self.parser.add_argument('--buffer_lines', dest='buffer_lines', type=int,
                                 help='Number of buffer lines')
        self.parser.add_argument('--overlap_lines', dest='overlap_lines', type=int,
                                 help='Number of overlapping lines')
        self.parser.add_argument('--use_phase_gradient_neutron', dest='use_phase', default=False,
                                 help='Phase gradient neutron flag')
        self.parser.add_argument('--use_intensity_neutron', dest='use_intensity', default=False,
                                 help='Intensity neutron flag')
        self.parser.add_argument('--phase_gradient_window', dest='pha_grad_win', default=5,
                                 help='Window size for phase gradient estimation')
        self.parser.add_argument('--neutron_phase_gradient_threshold', dest='neut_pha_grad_thr', default=3.0,
                                 help='Neutrons phase gradient threshold (radian)')
        self.parser.add_argument('--neutron_intensity_threshold', dest='neut_int_thr', default=8.0,
                                 help='Neutrons intensity threshold (sigma above mean)')
        self.parser.add_argument('--neutron_correlation_threshold', dest='neut_corr_thr', default=0.8,
                                 help='Maximum correlation for intensity neutrons')
        self.parser.add_argument('--trees_number', dest='trees_number', default=7,
                                 help='Number of tree realizations')
        self.parser.add_argument('--max_branch_length', dest='max_branch_length', default=64,
                                 help='Maximum branch length of trees')
        self.parser.add_argument('--spacing_ratio', dest='ratio_dxdy', default=1,
                                 help='Ratio of pixel spacings in X and Y directions')
        self.parser.add_argument('--initial_correlation_threshold', dest='init_corr_thr', default=0.1,
                                 help='Initial unwrapping correlation threshold')
        self.parser.add_argument('--max_correlation_threshold', dest='max_corr_thr', default=0.9,
                                 help='Maximum unwrap correlation threshold')
        self.parser.add_argument('--correlation_threshold_increments', dest='corr_incr_thr', default=0.1,
                                 help='Correlation threshold increments')
        self.parser.add_argument('-connected_component_area', dest='min_cc_area', default=0.003125,
                                 help='Minimum connected component size fraction of tile area')
        self.parser.add_argument('--bootstrap_lines', dest='num_bs_lines', default=16,
                                 help='Number of bootstrap lines')
        self.parser.add_argument('--min_overlap_area', dest='min_overlap_area', default=16,
                                 help='Minimum bootstrap overlapping area')
        self.parser.add_argument('--phase_variance_threshold', dest='pha_var_thr', default=8.0,
                                 help='Bootstrap phase variance threshold (radian)')
        self.parser.add_argument('--frequencies-polarizations', dest='freq_pols',
                                 action=StoreDictKeyPair, nargs='+', metavar="KEY=VAL",
                                 help='''Frequencies and polarizations to use.
                Key-values seperated by "=". Multiple polarizations comma seperated.
                Multiple key-values space seperated.''')

        self.args = argparse.Namespace()

    def parse(self):
        self.args = self.parser.parse_args()

        # check runconfig path if crossmul mode not enabled
        if self.args.run_config_path is not None:
            super().check_run_config_path()

        return self.args


# only used for unit tests
if __name__ == "__main__":
    unwrap_parser = UnwrapArgparse()
    unwrap_parser.parse()
