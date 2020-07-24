# -*- coding: utf-8 -*-

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
