# -*- coding: utf-8 -*-

import journal
import numpy as np
from numbers import Integral
from warnings import warn


def get_scalar_or_first(group, key, typeconv = lambda x: x):
    """
    Get a scalar from an HDF5 dataset that may be either a scalar or list

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object
    key : str
        Name of the dataset
    typeconv : function, optional
        Optional type conversion function to apply to result

    Returns
    -------
    value : object
        Scalar value corresponding to contents of a scalar dataset or the first
        entry in a vector dataset.
    """
    dset = group[key]
    if len(dset.shape) == 0:  # scalar
        value = group[key][()]
    else:
        value = group[key][0]
    return typeconv(value)


def parse_bool(s: np.bytes_) -> bool:
    """
    Parse a NISAR string representing a boolean value.

    Parameters
    ----------
    s : np.bytes_
        Byte string

    Returns
    -------
    bool
        True if s equals b'True' (any capitalization), False otherwise
    """
    return s.decode("utf-8").lower() == "true"


def get_list_from_scalar_or_list(group, key, typeconv = lambda x: x):
    """
    Get a list from an HDF5 dataset that may be either a scalar or list

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object
    key : str
        Name of the dataset
    typeconv : function, optional
        Optional type conversion function to apply to results

    Returns
    -------
    value : list
        List of values corresponding to contents of a scalar or vector dataset
    """
    dset = group[key]
    if len(dset.shape) == 0:  # scalar
        values = [group[key][()]]
    else:
        values = group[key][:]
    return [typeconv(value) for value in values]


class Identification(object):
    '''
    Simple object to hold identification information for NISAR products.
    '''
    def __init__(self, inobj, path='.', context=None):
        '''
        Identify given object as relevant NISAR product.
        '''

        self.missionId = None
        self.productType = None
        self.absoluteOrbitNumber = None
        self.lookDirection = None
        self.orbitPassDirection = None
        self.zdStartTime = None
        self.zdEndTime = None
        self.boundingPolygon = None
        self.listOfFrequencies = None
        self.diagnosticModeFlag = None
        self.diagnosticModeName = None

        ###Information from mission planning
        self.isJointObservation = None
        self.isUrgentObservation = None
        self.plannedDatatake = None
        self.plannedObservation = None



        import h5py

        #Any logging context
        if context is None:
            context = { 'info': journal.info('nisar.reader'),
                        'debug': journal.debug('nisar.reader'),
                        'error': journal.error('nisar.reader')}

        self.context = context

        #User has an open HDF5 file and is looking into it
        if isinstance(inobj, h5py.Group):
            self.unpack(inobj[path])
        #User provides HDF5 file and path inside it
        elif isinstance(inobj, str):
            with h5py.File(inobj, 'r') as fid:
                self.unpack(fid, path)


    def unpack(self, h5grp):
        '''
        Populate self with hdf5 group.
        '''
        from nisar.h5 import extractScalar, bytestring, extractWithIterator
        import isce3

        self.missionId = extractScalar(h5grp, 'missionId',
                                      bytestring, self.context['info'],
                                      'Mission could not be identified')
        self.productType = extractScalar(h5grp, 'productType',
                                      bytestring, self.context['error'],
                                      'Product type could not be determined')
        self.absoluteOrbitNumber = extractScalar(h5grp, 'absoluteOrbitNumber',
                                      int, self.context['info'],
                                      'Absolute orbit number could not be identified')
        self.lookDirection = extractScalar(h5grp, 'lookDirection',
                                      bytestring, self.context['error'],
                                      'Look direction could not be identified')
        self.orbitPassDirection = extractScalar(h5grp, 'orbitPassDirection',
                                      bytestring, self.context['info'],
                                      'Pass direction could not be identified')
        self.zdStartTime = extractScalar(h5grp, 'zeroDopplerStartTime',
                                      bytestring, self.context['error'])
        self.zdStartTime = isce3.core.DateTime(self.zdStartTime)

        self.zdEndTime = extractScalar(h5grp, 'zeroDopplerEndTime',
                                      bytestring, self.context['error'])
        self.zdEndTime = isce3.core.DateTime(self.zdEndTime)

        self.boundingPolygon = extractScalar(h5grp, 'boundingPolygon',
                                      bytestring, self.context['info'],
                                      'No bounding polygon could be identified')

        self.listOfFrequencies = extractWithIterator(h5grp, 'listOfFrequencies',
                                      bytestring, self.context['error'],
                                      'List of frequencies could not be determined')

        # Note that to avoid test failure for old-spec data products, the
        # field is directly extracted and evaluated to see if it holds a
        # proper value.
        # If old spec with scalar string value, it shall be either of
        # {"False", "True"}. "False" is for Science/DBF mode and "True is
        # for Multi-channel/DM2. A warning will be issued if otherwise.
        # This will avoid the unit test failure based on "winnipeg.h5".
        # For other obsolete data types such as numpy.ndarray(bool)
        # used in "envisat.h5", a warning will be issued to avoid test failure.
        # If new spec, the integer scalar value shall be either of {0, 1, 2}.
        # The final value will be an integer. In case of old/obsolete spec it
        # will be always set to 0 value, that is "Science" mode.
        self.diagnosticModeFlag = h5grp['diagnosticModeFlag'][()]

        # check for either another old sepc or new spec
        if isinstance(self.diagnosticModeFlag, Integral):
            # new spec assumes uint8 with either values {0, 1, 2}
            if self.diagnosticModeFlag not in range(3):
                raise ValueError('"diagnosticModeFlag" of new spec shall be'
                                 ' either of {0, 1, 2}!')
        elif isinstance(self.diagnosticModeFlag, bytes):
            # old spec assumes bytestring with either "True" or "False
            self.diagnosticModeFlag = self.diagnosticModeFlag.decode()
            # check str value and if it is Not "False", issue warning
            if self.diagnosticModeFlag == 'False':
                # assumes science/DBF mode
                self.diagnosticModeFlag = np.uint8(0)
            elif self.diagnosticModeFlag == 'True':
                # assumes Multi-channel DM2
                self.diagnosticModeFlag = np.uint8(2)
            else:
                # bad value, assumes science/DBF given it is not being used!
                self.diagnosticModeFlag = np.uint8(0)
                warn(
                    '"diagnosticModeFlag" for old spec with scalar string shall'
                    ' be either "False" or "True" rather than '
                    f'"{self.diagnosticModeFlag}"')
        else:
            # simply issue warning for other unsupported data types
            # and assume they are science mode.
            warn('The datatype "diagnosticModeFlag" is not supported!'
                 ' Either string (old spec) or integer (new spec) scalar!')
            self.diagnosticModeFlag = np.uint8(0)

        # provide a name for each flag as an extra attribute/info for clarity
        self.diagnosticModeName = {0: 'Science/DBF',
                                   1: 'Single-channel/DM1',
                                   2: 'Multi-channel/DM2'
                                   }.get(self.diagnosticModeFlag)

        ###Mission planning info
        # REE simulated L0B products have these as scalar when the spec says list
        self.plannedDatatake = get_list_from_scalar_or_list(h5grp,
            'plannedDatatakeId', bytestring)

        self.plannedObservation = get_list_from_scalar_or_list(h5grp,
            'plannedObservationId', bytestring)

        # spec changed from list to scalar, try to support both for now
        # Some old test data has bool instead of string, too.
        is_urgent = get_scalar_or_first(h5grp, "isUrgentObservation")
        if isinstance(is_urgent, np.bool_):
            warn("isUrgentObservation is boolean but expected string")
            self.isUrgentObservation = bool(is_urgent)
        else:
            self.isUrgentObservation = parse_bool(is_urgent)

        # isJointObservation was added in spec v1.2.0.  That's after the bool
        # type confusion was resolved, so just assume string.
        try:
            self.isJointObservation = parse_bool(h5grp["isJointObservation"][()])
        except KeyError:
            warn("Could not find isJointObservation in product identification.")
            # leave as None

        ###Processing type info to be added

# end of file
