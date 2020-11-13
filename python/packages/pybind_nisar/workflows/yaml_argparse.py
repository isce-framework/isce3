'''
class for parsing key value string into dict
class for parsing Yaml path from CLI
'''

import argparse
import os

import pybind_nisar.workflows.helpers as helpers

import journal


class StoreDictKeyPair(argparse.Action):
    '''
    Class that converts key value strings and parse into a dict.
    Intended for parsing frequency-polarizations strings.
    key and values seperated by '='
    values per key seperated by ','
    key-values pairs seperated by ' '
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
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('run_config_path', type=str, nargs='?',
                default='', help='Path to run config file')
        self.parser.add_argument('--no-log-file', dest='log_file', action='store_false',
                default=True, help='Disable logging to file. Log to file on by default.')
        self.args = argparse.Namespace()

    def parse(self):
        self.args = self.parser.parse_args()
        self.check_run_config_path()
        return self.args

    def check_run_config_path(self):
        error_channel = journal.error('YamlArgparse.check_run_config_path')

        # check args here instead of runconfig to start logging to file sooner
        if not os.path.isfile(self.args.run_config_path):
            err_str = f"{self.args.run_config_path} not a valid path"
            error_channel.log(err_str)
            raise FileNotFoundError(err_str)

        if self.args.log_file:
            helpers.check_log_dir_writable(self.args.run_config_path)

            # make a journal device that is attached to a file
            journal.debug.journal.device = "journal.file"
            journal.debug.journal.device.log = open(self.args.run_config_path + ".log", 'a')


# only used for unit tests
if __name__ == "__main__":
    yaml_parser = YamlArgparse()
    yaml_parser.parse()
