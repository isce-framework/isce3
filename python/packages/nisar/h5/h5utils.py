# -*- coding: utf-8 -*-
'''
This file contains utility methods for serializing / deserializing of basic data types from HDF5 files.
'''


bytestring = lambda x: x.decode('utf-8')


def extractScalar(h5grp, key, destType, logger=None, msg=None):
    '''
    Extract data of given type from HDF5 dataset and use logger for reporting.
    '''

    try:
        val = h5grp.get(key)[()]
        val = destType(val)
    except KeyError:
        if logger:
            logger.log('{0} not found at {1}'.format(key, h5grp.name))
    except:
        errmsg = 'Something went wrong when trying to parse {0}/{1} \n'.format(h5grp.name, key)
        if msg is not None:
            errmsg += msg
        raise ValueError(errmsg)

    return val


def extractWithIterator(h5grp, key, iterType, logger=None, msg=None):
    '''
    Extract data of given type from HDF5 dataset into a list and use logger for reporting.
    '''

    try:
        val = h5grp[key][()]
        val = [iterType(x) for x in val]
    except KeyError:
        if logger:
            logger.log('{0} not found at {1}'.format(key, h5grp.name))
    except:
        errmsg = 'Something went wrong when trying to parse {0}/{1} \n'.format(h5grp.name, key)
        if msg is not None:
            errmsg += msg
        raise ValueError(errmsg)

    return val

# end of file
