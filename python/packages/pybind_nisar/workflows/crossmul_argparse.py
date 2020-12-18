'''
class for parsing crossmul args from CLI
'''
import argparse

from pybind_nisar.workflows.yaml_argparse import StoreDictKeyPair, YamlArgparse


class CrossmulArgparse(YamlArgparse):
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
        self.parser.add_argument('--sec-raster', dest='sec_raster', type=str, default=None,
                help='Path to optional secondary SLC raster(s). If assigned, SLC(s) from secondary HDF5 will not \
                        be used. Can an directory with multiple rasters or HDF5.')
        self.parser.add_argument('--output-h5', dest='output_h5', type=str,
                help='Path the output dir')
        self.parser.add_argument('--flatten-path', dest='flatten_path', type=str,
                help='Path the directory containing range offset rasters.\
                No flattening applied if no argument provided.')
        self.parser.add_argument('--azimuth-looks', dest='azimuth_looks', type=int, default=1,
                help='Azimuth looks')
        self.parser.add_argument('--range-looks', dest='range_looks', type=int, default=1,
                help='Range looks')
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
    crossmul_parser = CrossmulArgparse()
    crossmul_parser.parse()
