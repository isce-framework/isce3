#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-

import os
from ruamel_yaml import YAML


def dictFromYaml(yaml_path):
    """
    reads runs params from YAML to dict
    """
    yaml = YAML()
    arg_dict = yaml.load(open(yaml_path, 'r'))
    return arg_dict


class RunArgs:

    def __init__(self, arg_dict):
        """
        generic class to store and transfer interferogram run params to workflows
        to be inherited workflow specific derived arg classes
        """

        # init required args shared across workflows to empty string
        self.reference = os.path.expanduser(arg_dict['reference'])
        self.product = os.path.expanduser(arg_dict['product'])
        self.slcpath = arg_dict['slcpath']
        self.frequency = arg_dict['frequency']
        self.polarization = arg_dict['polarization']
        self.dem = os.path.expanduser(arg_dict['dem'])
        self.alks = arg_dict['alks']
        self.rlks = arg_dict['rlks']
        self.workflow_outdir = arg_dict['outdir']
        self.gpu = arg_dict['gpu']

    def assignFromDict(self, a_dict, workflow_key):
        """
        copies dictionary contents to argument values
        """
        # check if non-default values exist
        if workflow_key in a_dict:
            # iterate through non-default values
            for k in a_dict[workflow_key]:
                # if key valid, then assign
                if k in list(self.__dict__):
                    setattr(self, k, a_dict[workflow_key][k])

class Rdr2GeoArgs(RunArgs):
    def __init__(self, arg_dict):
        # rdr2geo specific args with defaults
        super().__init__(arg_dict)
        self.product = self.reference
        self.outdir = os.path.join(self.workflow_outdir, 'rdr2geo')
        self.mask = ''
        self.assignFromDict(arg_dict, 'rdr2geo')

class Geo2RdrArgs(RunArgs):
    def __init__(self, arg_dict):
        # geo2rdr specific args with defaults
        super().__init__(arg_dict)
        self.topopath = 'rdr2geo/topo.vrt'
        self.azoff = 0.0
        self.rgoff = 0.0
        self.freq = self.frequency
        self.outdir = os.path.join(self.workflow_outdir, 'geo2rdr')
        self.assignFromDict(arg_dict, 'geo2rdr')

class ResampArgs(RunArgs):
    def __init__(self, arg_dict):
        # resamp specific args  with defaults
        super().__init__(arg_dict)
        self.outFilePath = os.path.join(self.workflow_outdir, 'resamp/product_secondary.slc')
        self.offsetdir = os.path.join(self.workflow_outdir, 'geo2rdr')
        self.linesPerTile = 0
        self.assignFromDict(arg_dict, 'resamp')

class CrossmulArgs(RunArgs):
    def __init__(self, arg_dict):
        # crossmul specific args  with defaults
        super().__init__(arg_dict)
        self.secondary = self.product
        self.secondaryRaster = os.path.join(self.workflow_outdir, 'resamp/product_secondary.slc')
        self.azband = 0.0
        self.rgoff = None
        self.cohFilePath = os.path.join(self.workflow_outdir, 'crossmul/crossmul.coh')
        self.intFilePath = os.path.join(self.workflow_outdir, 'crossmul/crossmul.int')
        self.assignFromDict(arg_dict, 'crossmul')

class GeocodeArgs(RunArgs):
    def __init__(self, arg_dict):
        # resamp specific args  with defaults
        super().__init__(arg_dict)
        # default: process interferogram since it's done regardless of multilook
        self.raster = os.path.join(self.workflow_outdir, 'crossmul/crossmul.int')
        self.h5 = self.reference
        self.outname = ''
        self.assignFromDict(arg_dict, 'geocode')


if __name__ == '__main__':
    """
    run to see if everything works
    """
    arg_dict = dictFromYaml('test.yaml')
    run_args = RunArgs(arg_dict)
    rdr2geo_args = Rdr2GeoArgs(arg_dict)
    geo2rdr_args = Geo2RdrArgs(arg_dict)
    resamp_args = ResampArgs(arg_dict)
    crossmul_args = CrossmulArgs(arg_dict)
    geocode_args = GeocodeArgs(arg_dict)

# end of file
