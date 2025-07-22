'''
class for parsing key value string into dict
class for parsing Yaml path from CLI
'''

import argparse
import os

import nisar.workflows.helpers as helpers

import journal


class StoreDictKeyPair(argparse.Action):
    '''
    Class that converts key value strings and parse into a dict.
    Intended for parsing frequency-polarizations strings.
    key and values separated by '='
    values per key separated by ','
    key-values pairs separated by ' '
    https://stackoverflow.com/a/42355279
    '''

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values[0].split():
            k, v = kv.split('=')
            v = v.split(',')
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


class YamlArgparse():
    '''
    Class for parsing Yaml path from CLI
    '''

    def __init__(self, resample_type = False):
        self.parser = argparse.ArgumentParser(description='',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('run_config_path', type=str, nargs='?',
                                 default=None, help='Path to run config file')
        self.parser.add_argument('--no-log-file', dest='log_file', action='store_false',
                                 default=True, help='Disable logging to file. Log to file on/True by default. If off/False, log file in runconfig yaml will not be read nor altered, log messasges will be sent to stdout, and all submodules will run i.e. no persistence checking.')
        self.parser.add_argument('--restart', action='store_true', default=False,
                                 help='Restart the InSAR workflow from the beginning and, if applicable, ignore the persistence state of previous run. Off/False by default.')
        if resample_type:
            self.parser.add_argument('--resample-type', dest='resample_type', default='coarse',
                                     help='Type of offsets (coregistered slc) to use in resample_slc (crossmul).\n'
                                          'Coarse: geometry offsets (secondary SLC resampled with geometry offsets)\n'
                                          'Fine: rubbersheeted dense offsets (secondary slc resampled with rubbersheet dense offsets')
        self.args = argparse.Namespace()

    def parse(self):
        '''
        Parse args from command line and then check argument validity.
        '''
        self.args = self.parser.parse_args()
        self.check_run_config_path()
        # Force restart if no log file provided
        # Otherwise persistence will attempt reading nonexistent file
        if not self.args.log_file:
            self.args.restart = True
        return self.args

    def check_run_config_path(self):
        '''
        Check command line argument validity.
        '''
        error_channel = journal.error('YamlArgparse.check_run_config_path')

        # check args here instead of runconfig to start logging to file sooner
        if not os.path.isfile(self.args.run_config_path):
            err_str = f"{self.args.run_config_path} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)


# only used for unit tests
if __name__ == "__main__":
    yaml_parser = YamlArgparse()
    yaml_parser.parse()
