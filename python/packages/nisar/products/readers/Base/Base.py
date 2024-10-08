# -*- coding: utf-8 -*-

import h5py
import os
import journal
import pyre
import isce3
from ..protocols import ProductReader


def get_hdf5_file_root_path(filename: str, root_path: str = None) -> str:
    '''
    Return root path from NISAR product (HDF5 file).

    Parameters
    ----------
    filename : str
        HDF5 filename
    root_path : str (optional)
        Preliminary root path to check before default root
        path list. This option is intended for non-standard NISAR products.

    Returns
    -------
    str
        Root path from HDF5 file

    '''

    error_channel = journal.error('get_hdf5_file_root_path')

    SCIENCE_PATH = '/science/'
    NISAR_SENSOR_LIST = ['SSAR', 'LSAR']
    with h5py.File(filename, 'r', libver='latest', swmr=True) as f:
        if root_path is not None and root_path in f:
            return root_path
        science_group = f[SCIENCE_PATH]
        for freq_band in NISAR_SENSOR_LIST:
            if freq_band not in science_group:
                continue
            return SCIENCE_PATH + freq_band

    error_msg = ("HDF5 could not find NISAR frequency"
                 f" band group LSAR or SSAR in file: {filename}")

    error_channel.log(error_msg)


def _join_paths(path1: str, path2: str) -> str:
    """Join two paths to be used in HDF5"""
    sep = '/'
    if path1.endswith(sep):
        sep = ''
    return path1 + sep + path2


class Base(pyre.component,
           family='nisar.productreader.base',
           implements=ProductReader):
    '''
    Base class for NISAR products.

    Contains common functionality that gets reused across products.
    '''
    _CFPath = pyre.properties.str(default='/')
    _CFPath.doc = 'Absolute path to scan for CF convention metadata'

    _RootPath = pyre.properties.str(default=None)
    _RootPath.doc = 'Absolute path to SAR data from L-SAR/S-SAR'

    _IdentificationPath = pyre.properties.str(default='identification')
    _IdentificationPath.doc = 'Absolute path to unique product identification information'

    _ProductType = pyre.properties.str(default=None)
    _ProductType.doc = 'The type of the product.'

    _MetadataPath = pyre.properties.str(default='metadata')
    _MetadataPath.doc = 'Relative path to metadata associated with standard product'

    _ProcessingInformation = pyre.properties.str(default='processingInformation')
    _ProcessingInformation.doc = 'Relative path to processing information associated with the product'

    _CalibrationInformation = pyre.properties.str(default='calibrationInformation')
    _CalibrationInformation.doc = 'Relative path to calibration information associated with the product'

    _SwathPath = pyre.properties.str(default='swaths')
    _SwathPath.doc = 'Relative path to swaths associated with standard product'

    _GridPath = pyre.properties.str(default='grids')
    _GridPath.doc = 'Relative path to grids associated with standard product'

    productValidationType = pyre.properties.str(default='BASE')
    productValidationType.doc = 'Validation tag to compare identification information against to ensure that the right product type is being used.'

    def __init__(self, hdf5file=None, **kwds):
        '''
        Constructor.
        '''

        # Set error channel
        self.error_channel = journal.error('Base')

        # Check hdf5file
        if hdf5file is None:
            err_str = f"Please provide an input HDF5 file"
            self.error_channel.log(err_str)

        # Filename
        self.filename = hdf5file

        # Identification information
        self.identification = None

        # Polarization dictionary
        self.polarizations = {}

        self.populateIdentification()

        self.identification.productType = self._ProductType

        if self._ProductType is None:
            return

        self.parsePolarizations()

    def _getFirstFrequency(self):
        '''
        Returns first available frequency
        '''
        if len(self.frequencies) == 0:
            error_channel = journal.error(
                'nisar.products.readers.Base._getFirstFrequency')
            error_msg = 'The product does not contain any frequency'
            error_channel.log(error_msg)
            raise RuntimeError(error_msg)
        return sorted(self.frequencies)[0]

    @pyre.export
    def getSwathMetadata(self, frequency=None):
        '''
        Returns metadata corresponding to given frequency.
        '''
        if frequency is None:
            frequency = self._getFirstFrequency()
        return isce3.product.Swath(self.filename, frequency)

    @pyre.export
    def getRadarGrid(self, frequency=None):
        '''
        Return radarGridParameters object
        '''
        if frequency is None:
            frequency = self._getFirstFrequency()
        return isce3.product.RadarGridParameters(self.filename, frequency)

    @pyre.export
    def getGridMetadata(self, frequency=None):
        '''
        Returns metadata corresponding to given frequency.
        '''
        if frequency is None:
            frequency = self._getFirstFrequency()
        return isce3.product.Grid(self.filename, frequency)

    @pyre.export
    def getOrbit(self):
        '''
        extracts orbit
        '''
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            orbitPath = os.path.join(self.MetadataPath, 'orbit')
            return isce3.core.load_orbit_from_h5_group(fid[orbitPath])

    @pyre.export
    def getAttitude(self):
        '''
        extracts attitude
        '''
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            attitudePath = _join_paths(self.MetadataPath, 'attitude')
            return isce3.core.Attitude.load_from_h5(fid[attitudePath])


    @pyre.export
    def getDopplerCentroid(self, frequency=None):
        '''
        Extract the Doppler centroid
        '''
        if frequency is None:
            frequency = self._getFirstFrequency()

        doppler_group_path = (f'{self.ProcessingInformationPath}/parameters/'
                              f'frequency{frequency}')

        # First, we look for the coordinate vectors `zeroDopplerTime`
        # and `slantRange` in the same level of the `dopplerCentroid` LUT.
        # If these vectors are not found, we look for the coordinate
        # vectors two levels below, following old RSLC specs.
        doppler_dataset_path = f'{doppler_group_path}/dopplerCentroid'
        zero_doppler_time_dataset_path = (f'{doppler_group_path}/'
                                          'zeroDopplerTime')
        slant_range_dataset_path = f'{doppler_group_path}/slantRange'

        zero_doppler_time_dataset_path_other = \
            f'{self.ProcessingInformationPath}/parameters/zeroDopplerTime'
        slant_range_dataset_path_other = (f'{self.ProcessingInformationPath}/'
                                          'parameters/slantRange')

        # extract the native Doppler dataset
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:

            if zero_doppler_time_dataset_path not in fid:
                zero_doppler_time_dataset_path = \
                    zero_doppler_time_dataset_path_other
            if slant_range_dataset_path not in fid:
                slant_range_dataset_path = \
                    slant_range_dataset_path_other

            doppler = fid[doppler_dataset_path][:]
            zeroDopplerTime = fid[zero_doppler_time_dataset_path][:]
            slantRange = fid[slant_range_dataset_path][:]

        dopplerCentroid = isce3.core.LUT2d(xcoord=slantRange,
                                           ycoord=zeroDopplerTime,
                                           data=doppler)

        return dopplerCentroid

    @pyre.export
    def getZeroDopplerTime(self):
        '''
        Extract the azimuth time of the zero Doppler grid
        '''

        zeroDopplerTimePath = os.path.join(self.SwathPath,
                                          'zeroDopplerTime')
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            zeroDopplerTime = fid[zeroDopplerTimePath][:]

        return zeroDopplerTime

    @pyre.export
    def getSlantRange(self, frequency=None):
        '''
        Extract the slant range of the zero Doppler grid
        '''
        if frequency is None:
            frequency = self._getFirstFrequency()
        slantRangePath = os.path.join(self.SwathPath,
                                      'frequency' + frequency, 'slantRange')

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            slantRange = fid[slantRangePath][:]

        return slantRange

    def parsePolarizations(self):
        '''
        Parse HDF5 and identify polarization channels available for each frequency.
        '''
        from nisar.h5 import bytestring, extractWithIterator

        try:
            frequencyList = self.frequencies
        except:
            raise RuntimeError('Cannot determine list of available frequencies without parsing Product Identification')

        ###Determine if product has swaths / grids
        if self.productType.startswith('G'):
            folder = self.GridPath
        else:
            folder = self.SwathPath

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for freq in frequencyList:
                root = os.path.join(folder, 'frequency{0}'.format(freq))
                polList = extractWithIterator(fid[root], 'listOfPolarizations', bytestring,
                                              msg='Could not determine polarization for frequency{0}'.format(freq))
                self.polarizations[freq] = [p.upper() for p in polList]

        return

    @property
    def CFPath(self):
        return self._CFPath

    @property
    def RootPath(self):
        if self._RootPath is None:
            self._RootPath = get_hdf5_file_root_path(
                self.filename)
        return self._RootPath

    @property
    def sarBand(self):
        """SAR band string such as 'L' or 'S' for NISAR."""
        return self.RootPath[-4]        

    @property
    def IdentificationPath(self):
        return os.path.join(self.RootPath, self._IdentificationPath)

    @property
    def ProductPath(self):
        return os.path.join(self.RootPath, self.productType)

    @property
    def MetadataPath(self):
        return os.path.join(self.ProductPath, self._MetadataPath)

    @property
    def ProcessingInformationPath(self):
        return os.path.join(self.MetadataPath, self._ProcessingInformation)

    @property
    def CalibrationInformationPath(self):
        return os.path.join(self.MetadataPath, self._CalibrationInformation)

    @property
    def SwathPath(self):
        return os.path.join(self.ProductPath, self._SwathPath)

    @property
    def GridPath(self):
        return os.path.join(self.ProductPath, self._GridPath)

    @property
    def productType(self):
        return self.identification.productType

    def populateIdentification(self):
        '''
        Read in the Identification information and assert identity.
        '''
        from .Identification import Identification

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fileID:
            h5grp = fileID[self.IdentificationPath]
            self.identification = Identification(h5grp)

    @property
    def frequencies(self):
        '''
        Return list of frequencies in the product.
        '''
        return self.identification.listOfFrequencies

    @staticmethod
    def validate(self, hdf5file):
        '''
        Validate a given HDF5 file.
        '''
        raise NotImplementedError

    def computeBoundingBox(self, epsg=4326):
        '''
        Compute the bounding box as a polygon in given projection system.
        '''
        raise NotImplementedError

    def getProductLevel(self):
        '''
        Returns the product level
        '''
        return "undefined"

# end of file
