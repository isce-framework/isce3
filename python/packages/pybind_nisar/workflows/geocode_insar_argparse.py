'''
class for parsing geocode args from CLI
'''
import argparse

from pybind_nisar.workflows.yaml_argparse import StoreDictKeyPair, YamlArgparse


class GeocodeInsarArgparse(YamlArgparse):
    '''
    Class for parsing geocode args from CLI.
    Inherits YAML parsing from YamlArgparse
    '''

    def __init__(self):
        super().__init__()

        self.parser.add_argument('--rslc-h5', dest='rslc_h5', type=str,
                                 help='Path to reference RSLC HDF5 used for '
                                      'interferogram formation. Needed for metadata.')
        self.parser.add_argument('--runw-h5', dest='runw_h5', type=str,
                                 help='Path to input RUNW HDF5 to be geocoded. '
                                      'Needed for unwrapped rasters.')
        self.parser.add_argument('--output-h5', dest='output_h5', type=str,
                                 help='Path the output GUNW HDF5')
        self.parser.add_argument('--dem', dest='dem', type=str,
                                 help='Path to DEM')
        self.parser.add_argument('--azimuth-looks', dest='azimuth_looks', type=int, default=1,
                                 help='Azimuth looks of the interferogram in the '
                                      'RUNW product. Must be odd integer.')
        self.parser.add_argument('--range-looks', dest='range_looks', type=int, default=1,
                                 help='Range looks of the interferogram '
                                      'in the RUNW product. Must be odd integer.')
        self.parser.add_argument('--interp', dest='interp_method', type=str, default='BILINEAR',
                                 help='DEM interpolation method. Available methods '
                                      'BILINEAR (default), BICUBIC, NEAREST and BIQUINTIC.')
        self.parser.add_argument('--no-connected-components', action='store_true', default=False,
                                 help='Disable geocoded connectedComponents '
                                      'dataset. False by default.')
        self.parser.add_argument('--no-coherence', action='store_true', default=False,
                                 help='Disable geocoded coherenceMagnitude '
                                      'dataset. False by default.')
        self.parser.add_argument('--no-unwrapped-phase', action='store_true', default=False,
                                 help='Disable geocoded unwrappedPhase '
                                      'dataset. False by default.')
        self.parser.add_argument('--frequencies-polarizations', dest='freq_pols',
                                 action=StoreDictKeyPair, nargs='+', metavar="KEY=VAL",
                                 help='''Frequencies and polarizations to use.
                                         Key-values seperated by "=". Multiple polarizations comma seperated.
                                         Multiple key-values space seperated.''')

        self.args = argparse.Namespace()

    def parse(self):
        self.args = self.parser.parse_args()

        # check runconfig path if geocode mode not enabled
        if self.args.run_config_path is not None:
            super().check_run_config_path()

        return self.args


# only used for unit tests
if __name__ == "__main__":
    geocode_parser = GeocodeInsarArgparse()
    geocode_parser.parse()
