# -*- coding: utf-8 -*-

from logging import error
import os
import h5py
import pyre
import journal
from ..Base import Base
from .Identification import Identification
from nisar.products.readers.GenericProduct import \
    get_hdf5_file_product_type

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
        ###Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('SLC')

        self.identification.productType = \
            get_hdf5_file_product_type(
                self.filename,
                product_type = self.productType,
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


