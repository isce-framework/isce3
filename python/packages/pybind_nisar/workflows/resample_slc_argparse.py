'''
class for parsing resampSlc from CLI
'''

import argparse

from pybind_nisar.workflows.yaml_argparse import StoreDictKeyPair, YamlArgparse


class ResampleSlcArgparse(YamlArgparse):
    '''
    Class for parsing resample_slc args from CLI
    Inherits YAML parsing from YamlArgparse
    '''

    def __init__(self):
        super().__init__()

        self.parser.add_argument('--input-h5', dest='input_h5', type=str, default='',
                                 help='Path to h5 to resample')
        self.parser.add_argument('--offset-dir', dest='off_dir', type=str, default=None,
                                 help='''
                                 Path to input offset directory. Input will either be resample_slc/freqA or
                                 resample_slc/freqB within offset dir path.
                                 ''')
        self.parser.add_argument('--scratch', dest='scratch', type=str, default='',
                                 help='''
                                 Path to output directory. If no offset dir, scratch will be also used as
                                 offset directory path. Output will either be in geo2rdr/freqA or geo2rdr/freqB
                                 within the scratch path
                                 ''')
        self.parser.add_argument('--linesPerTile', dest='lines_per_tile', type=int, default=0,
                                 help='Number of resampled lines per iteration')
        self.parser.add_argument('--frequencies-polarizations', dest='freq_pols',
                                 action=StoreDictKeyPair, nargs='+', metavar="KEY=VAL", default={},
                                 help='''Frequencies and polarizations to process.
                                 Key-values separated by "=". Multiple polarizations comma separated.
                                 Multiple key-values  space separated.''')

        self.args = argparse.Namespace()

    def parse(self):
        self.args = self.parser.parse_args()

        # Check runconfig path if resample_slc mode is not enabled
        if self.args.run_config_path is not None:
            super().check_run_config_path()

        return self.args


# Only used for unit test
if __name__ == "__main__":
    resample_slc_parser = ResampleSlcArgparse()
    resample_slc_parser.parse()
