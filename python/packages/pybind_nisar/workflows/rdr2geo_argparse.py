'''
class for parsing rdr2geo args from CLI
'''
import argparse

from pybind_nisar.workflows.yaml_argparse import StoreDictKeyPair, YamlArgparse


class Rdr2geoArgparse(YamlArgparse):
    '''
    Class for parsing rdr2geo args from CLI.
    Inherits YAML parsing from YamlArgparse
    '''

    def __init__(self):
        super().__init__()

        self.parser.add_argument('--input-h5', dest='input_h5', type=str, default='',
                                 help='Path to input h5')
        self.parser.add_argument('--dem', dest='dem', type=str, default='',
                                 help='Path to DEM')
        self.parser.add_argument('--scratch', dest='scratch', type=str, default='',
                                 help='''
                                 Path to ouptput directory. Data will written to either rdr2geo/freqA or
                                 rdr2geo/freqB directory within the scratch path.
                                ''')
        self.parser.add_argument('--frequencies-polarizations', dest='freq_pols',
                                 action=StoreDictKeyPair, nargs='+', metavar="KEY=VAL", default={},
                                 help='''Frequencies and polarizations to use.
                                 Key-values seperated by "=". Multiple polarizations comma seperated.
                                 Multiple key-values space seperated.''')

        self.args = argparse.Namespace

    def parse(self):
        self.args = self.parser.parse_args()

        # check runconfig path if rdr2geo mode not enabled
        if self.args.run_config_path is not None:
            super().check_run_config_path()

        return self.args


# only used for unit tests
if __name__ == "__main__":
    rdr2geo_parser = Rdr2geoArgparse()
    rdr2geo_parser.parse()
