# -*- coding: utf-8 -*-

from logging import error
import os
import h5py
import pyre
import journal
import numpy as np
from nisar.products.readers.Base import (Base,
                                         get_hdf5_file_root_path)


def open_product(filename: str, root_path: str = None):
    '''
    Open NISAR product (HDF5 file), instantianting an object
    of an existing product class (e.g. RSLC, RRSD), if
    defined, or a general product (GeneralProduct) otherwise.

    Parameters
    ----------
    filename : str
        HDF5 filename
    root_path : str (optional)
        Preliminary root path to check before default root
        path list. This option is intended for non-standard products.

    Returns
    -------
    object
        Object derived from the base class

    '''

    if root_path is None:
        root_path = get_hdf5_file_root_path(filename, root_path = root_path)

    product_type = get_hdf5_file_product_type(filename, root_path = root_path)

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

    elif (product_type in ['GCOV', 'GSLC', 'GUNW', 'GOFF']):
        # return GenericL2Product obj
        from nisar.products.readers.GenericProduct \
            import GenericL2Product
        kwargs['_ProductType'] = product_type
        return GenericL2Product(**kwargs)

    kwargs['_ProductType'] = product_type

    # return ProductFactory obj
    return GenericProduct(**kwargs)


def get_hdf5_file_product_type(filename: str, root_path: str = None) -> str:
    '''
    Return product type from NISAR product (HDF5 file).

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
        Product type
    '''
    if root_path is None:
        root_path = get_hdf5_file_root_path(filename, root_path=root_path)

    with h5py.File(filename, 'r', libver='latest', swmr=True) as f:
        product_type_ds = f[root_path+'/identification/productType']
        product_type = str(np.asarray(product_type_ds, dtype=str))

        # The product group name should be "RSLC" per the spec.
        # However, early sample products used "SLC" instead.
        # We maintain compatibility with both options.

        if product_type == 'SLC':
            return 'RSLC'
        return product_type


class GenericProduct(Base, family='nisar.productreader.product'):
    '''
    Class for parsing NISAR products into isce3 structures.
    '''

    def __init__(self, **kwds):
        '''
        Constructor to initialize product with HDF5 file.
        '''

        # Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('GenericProduct')

        self.identification.productType = \
            get_hdf5_file_product_type(self.filename,
                                       root_path = self.RootPath)

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

    def getProductLevel(self):
        '''
        Returns the product level
        '''
        if self.productType in ['GCOV', 'GSLC', 'GUNW', 'GOFF']:
            return "L2"
        if self.productType in ['RSLC', 'RIFG', 'RUNW', 'ROFF']:
            return "L1"
        if self.productType in ['RRSD']:
            return "L0B"
        return "undefined"