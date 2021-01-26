# -*- coding: utf-8 -*-

import h5py
import pyre
from ..Base import Base
from .Identification import Identification

class SLC(Base, family='nisar.productreader.slc'):
    '''
    Class for parsing NISAR SLC products into isce structures.
    '''
   
    productValidationType = pyre.properties.str(default='SLC')
    productValidationType.doc = 'Validation tag to ensure correct product type'

    def __init__(self, **kwds):
        '''
        Constructor to initialize product with HDF5 file.
        '''

        ###Read base product information like Identification
        super().__init__(**kwds) 


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
