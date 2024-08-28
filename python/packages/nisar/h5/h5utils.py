# -*- coding: utf-8 -*-
'''
This file contains utility methods for serializing / deserializing of basic data types from HDF5 files.
'''
import h5py
import numpy as np


bytestring = lambda x: x.decode('utf-8')


def extractScalar(h5grp, key, destType, logger=None, msg=None):
    '''
    Extract data of given type from HDF5 dataset and use logger for reporting.
    '''
    def handle_message(err: str):
        """Append input message and log it if possible.
        """
        errmsg = err if msg is None else err + msg
        if logger is not None:
            logger.log(errmsg)
        return errmsg

    try:
        val = h5grp[key][()]
        val = destType(val)
    except KeyError as e:
        raise KeyError(handle_message(f'{key} not found at {h5grp.name}')) from e
    except Exception as e:
        errmsg = 'Something went wrong when trying to parse {0}/{1} \n'.format(h5grp.name, key)
        raise ValueError(handle_message(errmsg)) from e

    return val


def extractWithIterator(h5grp, key, iterType, logger=None, msg=None):
    '''
    Extract data of given type from HDF5 dataset into a list and use logger for reporting.
    '''
    try:
        val = h5grp[key][()]
        val = [iterType(x) for x in val]
    except KeyError:
        errmsg = '{0} not found at {1}'.format(key, h5grp.name)
        if logger:
            logger.log(errmsg)
        raise KeyError(errmsg)
    except:
        errmsg = 'Something went wrong when trying to parse {0}/{1} \n'.format(h5grp.name, key)
        if msg is not None:
            errmsg += msg
        if logger:
            logger.log(errmsg)
        raise ValueError(errmsg)

    return val


def set_string(group: h5py.Group, name: str, data: str) -> h5py.Dataset:
    "Simplify updates of fixed-length strings."
    if name in group:
        del group[name]
    return group.create_dataset(name, data=np.bytes_(data))
