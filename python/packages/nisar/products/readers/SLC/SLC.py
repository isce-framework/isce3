# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re

import h5py
import journal
import pyre
from nisar.products.readers.GenericProduct import get_hdf5_file_product_type
from isce3.core.types import ComplexFloat16Decoder, complex32

from ..Base import Base
from .Identification import Identification

PRODUCT = 'RSLC'


def is_complex32(dataset: h5py.Dataset) -> bool:
    '''
    Check if the input dataset is complex32 (i.e. pairs of 16-bit floats).

    Parameters
    ----------
    dataset : h5py.Dataset
        The input dataset.

    Returns
    -------
    bool
        True if the input dataset is complex32; otherwise False.
    '''
    # h5py 3.8.0 returns a compound datatype when accessing a complex32
    # dataset's dtype (https://github.com/h5py/h5py/pull/2157). Previous
    # versions of h5py raise TypeError when attempting to get the dtype. In this
    # case, we try to infer whether the dataset was complex32 based on the error
    # message.
    try:
        dtype = dataset.dtype
    except TypeError as e:
        regex = re.compile(r"^data type '([<>|=])?c4' not understood$")
        errmsg = str(e)
        if regex.match(errmsg):
            return True
        else:
            raise
    else:
        return dtype == complex32


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

    def getSlcDatasetAsNativeComplex(
        self, frequency: str, polarization: str
    ) -> h5py.Dataset | ComplexFloat16Decoder:
        '''
        Get an SLC raster layer as a complex64 or complex128 dataset.

        Return the SLC dataset corresponding to a given frequency sub-band and
        polarization from the HDF5 file as a complex64 (i.e. pairs of 32-bit floats)
        or complex128 (i.e. pairs of 64-bit floats) dataset. If the data was stored as
        complex32 (i.e. pairs of 16-bit floats), it will be lazily converted to
        complex64 when accessed.

        Parameters
        ----------
        frequency : str
            The frequency sub-band of the SLC dataset.
        pol : str
            The Tx and Rx polarization of the SLC dataset.

        Returns
        -------
        h5py.Dataset or isce3.core.types.ComplexFloat16Decoder
            The HDF5 dataset, possibly wrapped in a decoder layer that handles
            converting from half precision complex values to single precision.
        '''
        dataset = self.getSlcDataset(frequency, polarization)

        if is_complex32(dataset):
            return ComplexFloat16Decoder(dataset)
        else:
            return dataset

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

    def is_dataset_complex32(self, freq, pol):
        '''
        Determine if RSLC raster is of data type complex32

        Parameters
        ----------
        freq: str
            Frequency of raster to check
        pol: str
            Polarization of raster to check
        '''
        # Set error channel
        error_channel = journal.error('SLC.is_dataset_complex32')

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

            return is_complex32(h[slc_path])
