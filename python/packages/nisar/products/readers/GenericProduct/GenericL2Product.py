# -*- coding: utf-8 -*-

from logging import error
import os
import h5py
import pyre
import journal
import isce3
from nisar.products.readers.Base import get_hdf5_file_root_path
from nisar.products.readers.GenericProduct import GenericProduct, \
        get_hdf5_file_product_type


class GenericL2Product(GenericProduct, family='nisar.productreader.product'):
    '''
    Class for parsing NISAR L2 products into isce3 structures.
    '''

    def __init__(self, **kwds):
        '''
        Constructor to initialize product with HDF5 file.
        '''

        ###Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('GenericL2Product')

        self.identification.productType = \
            get_hdf5_file_product_type(self.filename,
                                       root_path = self.RootPath)

        NISAR_L2_PRODUCT_LIST = ['GCOV', 'GSLC', 'GUNW', 'GOFF']
        if self.identification.productType not in NISAR_L2_PRODUCT_LIST:
            error_msg = (f'ERROR input HDF5 file {self.filename} is not a'
                         ' valid NISAR L2 product.')
            error_channel = journal.error('GenericL2Product')
            error_channel.log(error_msg)
            raise RuntimeError(error_msg)

        self.parsePolarizations()


    def parsePolarizations(self):
        '''
        Parse HDF5 and identify polarization channels available for each frequency.
        '''
        from nisar.h5 import bytestring, extractWithIterator

        try:
            frequencyList = self.frequencies
        except:

            error_msg = ('Cannot determine list of available frequencies'
                         ' without parsing Product Identification')
            self.error_channel.log(error_msg)
            raise RuntimeError(error_msg)

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for freq in frequencyList:
                root = os.path.join(self.GridPath, 'frequency{0}'.format(freq))
                if root not in fid:
                    continue
                polList = extractWithIterator(
                    fid[root], 'listOfPolarizations', bytestring,
                    msg='Could not determine polarization for frequency{0}'.format(freq))
                self.polarizations[freq] = [p.upper() for p in polList]

    def getGeoGridProduct(self):
        '''
        Returns metadata corresponding to given frequency.
        '''
        return isce3.product.GeoGridProduct(self.filename)

    def getProductLevel(self):
        '''
        Returns the product level
        '''
        return "L2"

    @pyre.export
    def getDopplerCentroid(self, frequency=None):
        '''
        Extract the Doppler centroid
        '''
        if frequency is None:
            frequency = self._getFirstFrequency()

        doppler_group_path = (
            f'{self.MetadataPath}/sourceData/processingInformation/parameters/'
            f'frequency{frequency}')

        doppler_dataset_path = f'{doppler_group_path}/dopplerCentroid'
        zero_doppler_time_dataset_path = (f'{doppler_group_path}/'
                                          'zeroDopplerTime')
        slant_range_dataset_path = f'{doppler_group_path}/slantRange'

        # extract the native Doppler dataset
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:

            doppler = fid[doppler_dataset_path][:]
            zeroDopplerTime = fid[zero_doppler_time_dataset_path][:]
            slantRange = fid[slant_range_dataset_path][:]

        dopplerCentroid = isce3.core.LUT2d(xcoord=slantRange,
                                           ycoord=zeroDopplerTime,
                                           data=doppler)

        return dopplerCentroid
