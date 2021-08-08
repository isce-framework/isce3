'''
class for parsing crossmul args from CLI
'''
import argparse

from nisar.workflows.yaml_argparse import StoreDictKeyPair, YamlArgparse


class BandpassArgparse(YamlArgparse):
    '''
    Class for parsing crossmul args from CLI.
    Inherits YAML parsing from YamlArgparse
    '''

    def __init__(self):
        super().__init__()

        self.parser.add_argument('--ref-hdf5', dest='ref_hdf5', type=str,
                                 help='Path to reference HDF5')
        self.parser.add_argument('--sec-hdf5', dest='sec_hdf5', type=str,
                                 help='Path to secondary HDF5. Provides metadata and, if secondary raster not assigned, \
                                       coregistered secondary SLC(s).')
        self.parser.add_argument('--rows_per_block', dest='rows_per_block', type=int, default=1000,
                                 help='Rows per block per SLCs to be processed at a time')
        self.parser.add_argument('--frequencies-polarizations', dest='freq_pols',
                                 action=StoreDictKeyPair, nargs='+', metavar="KEY=VAL",
                                 help='''Frequencies and polarizations to use.
                                         Key-values seperated by "=". Multiple polarizations comma seperated.
                                         Multiple key-values space seperated.''')

        self.args = argparse.Namespace()

    def parse(self):
        self.args = self.parser.parse_args()

        if self.args.run_config_path is not None:
            super().check_run_config_path()

        return self.args


# only used for unit tests
if __name__ == "__main__":
    bandpass_parser = BandpassArgparse()
    bandpass_parser.parse()
