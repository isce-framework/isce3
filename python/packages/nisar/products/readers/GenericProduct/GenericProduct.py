# -*- coding: utf-8 -*-

from logging import error
import os
import h5py
import pyre
import journal
from nisar.products.readers.Base import (Base,
                                        get_hdf5_file_root_path)


def open_product(filename: str,
                 product_type: str = None,
                 root_path: str = None):
    '''
    Open NISAR product (HDF5 file), instantianting an object
    of an existing product class (e.g. RSLC, RRSD), if
    defined, or a general product (GeneralProduct) otherwise.

    Parameters
    ----------
    filename : str
        HDF5 filename
    product_type : str
        Preliminary product type to check (e.g. RCOV) before default product type list
    root_path : str
        Preliminary root path to check (e.g., XSAR, PSAR) before default root
        path list

    Returns
    -------
    object
        Object derived from the base class

    '''

    if root_path is None:
        root_path = get_hdf5_file_root_path(
            filename, root_path = root_path)
    product_type = get_hdf5_file_product_type(
        filename, product_type = product_type,
        root_path = root_path)

    # set keyword arguments for class constructors
    kwargs = {}
    kwargs['hdf5file'] = filename
    kwargs['_RootPath'] = root_path

    if (product_type == 'RSLC'):

        # return SLC obj
        from nisar.products.readers import SLC
        return SLC(**kwargs)
    elif (product_type == 'RRSD'):

        # return Raw obj
        from nisar.products.readers.Raw import Raw
        return Raw(**kwargs)

    kwargs['_ProductType'] = product_type

    # return ProductFactory obj
    return GenericProduct(**kwargs)


def get_hdf5_file_product_type(filename,
                               product_type = None,
                               root_path = None):
    '''
    Return product type from HDF5 file. If a product type is
    provided as a parameter and it exists, it will have precedence
    over the product type from the HDF5 file

    Parameters
    ----------
    filename : str
        HDF5 filename
    product_type : str
        Preliminary product type to check (e.g. RCOV) before default product type list
    root_path : str
        Preliminary root path to check (e.g., XSAR, PSAR) before default root
        path list

    Returns
    -------
    str
        Product type
    '''
    # The product group name should be "RSLC" per the spec.
    # However, early sample products used "SLC" instead.
    # We maintain compatibility with both options.

    error_channel = journal.error('get_hdf5_file_product_type')

    if root_path is None:
        root_path = get_hdf5_file_root_path(
            filename, root_path=root_path)
    NISAR_PRODUCT_LIST = ['RRSD', 'RSLC', 'SLC', 'RIFG', 'RUNW',
                          'GCOV', 'GSLC', 'GUNW']
    with h5py.File(filename, 'r', libver='latest', swmr=True) as f:
        g = f[root_path]
        if product_type is not None and product_type in g:
            return product_type
        for product in NISAR_PRODUCT_LIST:
            if product not in g:
                continue
            if product in ['SLC', 'RSLC']:
                return 'RSLC'
            return product

    error_msg = ("HDF5 could not find NISAR product group"
                 f" in file: {filename}")

    error_channel.log(error_msg)


class GenericProduct(Base, family='nisar.productreader.product'):
    '''
    Class for parsing NISAR products into isce3 structures.
    '''

    def __init__(self, **kwds):
        '''
        Constructor to initialize product with HDF5 file.
        '''

        ###Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('GenericProduct')

        self.identification.productType = \
            get_hdf5_file_product_type(
                self.filename,
                product_type = self.productType,
                root_path = self.RootPath)

        if (self.productValidationType != 'BASE' and
                self.productValidationType != self.productType):
            self.error_channel.log(
                f'Expecting product {self.productValidationType}'
                f' but {self.productType} found')

        self.parsePolarizations()


    def parsePolarizations(self):
        '''
        Parse HDF5 and identify polarization channels available for each frequency.
        '''
        from nisar.h5 import bytestring, extractWithIterator

        try:
            frequencyList = self.frequencies
        except:
            self.error_channel.log(
                'Cannot determine list of available frequencies'
                ' without parsing Product Identification')

        ###Determine if product has swaths / grids

        folder_list = [self.SwathPath, self.GridPath]

        flag_found_folder = False
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for folder in folder_list:
                for freq in frequencyList:
                    root = os.path.join(folder, 'frequency{0}'.format(freq))
                    if root not in fid:
                        continue
                    flag_found_folder = True
                    polList = extractWithIterator(
                        fid[root], 'listOfPolarizations', bytestring,
                        msg='Could not determine polarization for frequency{0}'.format(freq))
                    self.polarizations[freq] = [p.upper() for p in polList]
                if flag_found_folder:
                    break
