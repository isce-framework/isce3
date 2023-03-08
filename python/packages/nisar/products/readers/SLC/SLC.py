# -*- coding: utf-8 -*-

import os

import h5py
import journal
import numpy as np
import pyre
from nisar.products.readers.GenericProduct import get_hdf5_file_product_type
from nisar.types import complex32

from ..Base import Base
from .Identification import Identification

PRODUCT = 'RSLC'

class SLC(Base, family='nisar.productreader.slc'):
    '''
    Class for parsing NISAR SLC products into isce structures.
    '''

    productValidationType = pyre.properties.str(default=PRODUCT)
    productValidationType.doc = 'Validation tag to ensure correct product type'

    _ProductType = pyre.properties.str(default=PRODUCT)
    _ProductType.doc = 'The type of the product.'

    def __init__(self, **kwds):
        '''
        Constructor to initialize product with HDF5 file.
        '''
        # Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('SLC')

        self.identification.productType = \
            get_hdf5_file_product_type(
                self.filename,
                root_path = self.RootPath)

        if (self.productType != self.productValidationType):
            self.error_channel.log(
                f'Expecting product {self.productValidationType}'
                f' but product {self.productType} was provided')

    def populateIdentification(self):
        '''
        Read in the Identification information and assert identity.
        '''
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            h5grp = f[self.IdentificationPath]
            self.identification = Identification(h5grp)

    @property
    def ProductPath(self):
        # The product group name should be "RSLC" per the spec. However, early
        # sample products used "SLC" instead, and identification.productType is
        # not reliable, either. We maintain compatibility with both options.
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            g = f[self.RootPath]
            if "RSLC" in g:
                return f"{g.name}/RSLC"
            elif "SLC" in g:
                return f"{g.name}/SLC"
        raise RuntimeError("HDF5 file missing 'RSLC' or 'SLC' product group.")

    def getSlcDataset(self, frequency, polarization):
        '''
        Return SLC dataset of given frequency and polarization from hdf5 file
        '''

        # TODO add checks for (1) file open error (2) path check
        slcDataset = None

        # open H5 with swmr mode enabled
        fid = h5py.File(self.filename, 'r', libver='latest', swmr=True)

        # build path the desired dataset
        ds_path = self.slcPath(frequency, polarization)

        # get dataset
        slcDataset = fid[ds_path]

        # return dataset
        return slcDataset

    def slcPath(self, frequency, polarization):
        '''
        return path to hdf5 dataset of given frequency and polarization
        '''
        dataset_path = os.path.join(self.SwathPath, f'frequency{frequency}', polarization)
        return dataset_path

    def getProductLevel(self):
        '''
        Returns the product level
        '''
        return "L1"


    def is_dataset_complex64(self, freq, pol):
        '''
        Determine if RSLC raster is of data type complex64

        Parameters
        ----------
        freq: str
            Frequency of raster to check
        pol: str
            Polarization of raster to check
        '''
        # Set error channel
        error_channel = journal.error('SLC.is_dataset_complex64')

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as h:
            freq_path = f'/{self.SwathPath}/frequency{freq}'
            if freq_path not in h:
                err_str = f'Frequency {freq} not found in SLC'
                error_channel.log(err_str)
                raise LookupError(err_str)

            slc_path = self.slcPath(freq, pol)
            if slc_path not in h:
                err_str = f'Polarization {pol} for frequency {freq} not found in SLC'
                error_channel.log(err_str)
                raise LookupError(err_str)

            # h5py 3.8.0 returns a compound datatype when accessing a complex32
            # dataset's dtype (https://github.com/h5py/h5py/pull/2157).
            # Previous versions of h5py raise TypeError when attempting to
            # get the dtype. If such exception was raised, we assume the
            # datatype was complex32
            try:
                dtype = h[slc_path].dtype
            except TypeError:
                return False
            else:
                return dtype == np.complex64
