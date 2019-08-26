# -*- coding: utf-8 -*-
#
# Authors: ?, Liang Yu
# Copyright 2019-
#

import pyre
from ..Base import Base

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


# end of file
