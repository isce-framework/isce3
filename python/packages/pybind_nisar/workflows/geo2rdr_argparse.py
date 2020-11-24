'''
class for parsing geo2rdr args from CLI
'''

import argparse

from pybind_nisar.workflows.yaml_argparse import StoreDictKeyPair, YamlArgparse


class Geo2rdrArgparse(YamlArgparse):
    '''
    Class for parsing geo2rdr args from CLI
    Inherits YAML parsing from YamlArgparse
    '''

    def __init__(self):
        super().__init__()

        self.parser.add_argument('--input-h5', dest='input_h5', type=str, default='',
                                 help='Path to input h5')
        self.parser.add_argument('--topo', dest='topo', type=str, default=None,
                                 help='''
                                 Path to input topo/rdr2geo directory. Input will either be in
                                 rdr2geo/freqA or rdr2geo/freqB directory within the topo path.
                                 ''')
        self.parser.add_argument('--dem', dest='dem', type=str, default='',
                                 help='Path to DEM')
        self.parser.add_argument('--max-iter', dest='max_iter', type=int, default=25,
                                 help='Maximum number of iterations')
        self.parser.add_argument('--threshold', dest='threshold', type=float, default=1e-8,
                                 help='Convergence threshold')
        self.parser.add_argument('--scratch', dest='scratch', type=str, default='',
                                 help='''
                                 Path to output directory. If no topo path given, scratch will
                                 also be used as the input topo path. Output will either be in
                                 rdr2geo/freqA or rdr2geo/freqB directory within the scratch path.
                                 ''')
        self.parser.add_argument('--frequencies-polarizations', dest='freq_pols',
                                 action=StoreDictKeyPair, nargs='+', metavar="KEY=VAL", default={},
                                 help='''Frequencies and polarization to process.
                                 Key-values separated by "=". Multiple polarizations comma separated.
                                 Multiple key-values space separated.''')

        self.args = argparse.Namespace()

    def parse(self):
        self.args = self.parser.parse_args()

        # Check runconfig path if geo2rdr mode is not enabled
        if self.args.run_config_path is not None:
            super().check_run_config_path()

        return self.args


# Only used for unit tests
if __name__ == "__main__":
    geo2rdr_parser = Geo2rdrArgparse()
    geo2rdr_parser.parse()
