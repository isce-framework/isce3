# -*- coding: utf-8 -*-

import pyre
from ..protocols import ProductReader

class Base(pyre.component,
           family='nisar.productreader.base',
           implements=ProductReader):
    '''
    Base class for NISAR products.

    Contains common functionality that gets reused across products.
    '''
    _CFPath = pyre.properties.str(default='/')
    _CFPath.doc = 'Absolute path to scan for CF convention metadata'

    _RootPath = pyre.properties.str(default='/science/LSAR')
    _RootPath.doc = 'Absolute path to SAR data from L-SAR/S-SAR'

    _IdentificationPath = pyre.properties.str(default='identification')
    _IdentificationPath.doc = 'Absolute path ath to unique product identification information'

    _ProductType = pyre.properties.str(default='SLC')
    _ProductType.doc = 'The type of the product.'

    _MetadataPath = pyre.properties.str(default='metadata')
    _MetadataPath.doc = 'Relative path to metadata associated with standard product'

    _ProcessingInformation = pyre.properties.str(default='processingInformation')
    _ProcessingInformation.doc = 'Relative path to processing information associated with the product'

    _SwathPath = pyre.properties.str(default='swaths')
    _SwathPath.doc = 'Relative path to swaths associated with standard product'

    _GridPath = pyre.properties.str(default='grids')
    _GridPath.doc = 'Relative path to grids associated with standard product'

    productValidationType = pyre.properties.str(default='BASE')
    productValidationType.doc = 'Validation tag to compare identification information against to ensure that the right product type is being used.'

    def __init__(self, **kwds):
        '''
        Constructor.
        '''
        # Filename
        self.filename = None

        # Identification information
        self.identification = None

        # Polarization dictionary
        self.polarizations = {}
    
    @pyre.export
    def parse(self, hdf5file):
        '''
        Parse the HDF5 file and populate ISCE data structures.
        '''
        import h5py

        self.filename = hdf5file
        self.populateIdentification()
        #For now, Needs to be an assertion check in the future
        self.identification.productType = self.productValidationType
        self.parsePolarizations()

    @pyre.export
    def getSwathMetadata(self, frequency):
        '''
        Returns metadata corresponding to given frequency.
        '''
        import h5py
        import isce3
        import os
        #swathPath = os.path.join(self._RootPath, self.productType,
        #                          self._SwathPath)
        swath = isce3.product.swath()
        with h5py.File(self.filename, 'r') as fid:
              #swathGrp = fid[swathPath]
              swathGrp = fid[self.SwathPath]
              swath.loadFromH5(swathGrp, frequency)

        return swath

    @pyre.export
    def getGridMetadata(self, frequency):
        '''
        Returns metadata corresponding to given frequency.
        '''
        raise NotImplementedError

    @pyre.export
    def getOrbit(self):
        '''
        extracts orbit 
        '''
        import h5py
        import isce3
        import os

        #orbitPath = os.path.join(self._RootPath, self.productType, 
        #        self._MetadataPath, 'orbit')
        orbitPath = os.path.join(self.MetadataPath, 'orbit')

        orbit = isce3.core.orbit()
        with h5py.File(self.filename, 'r') as fid:
            orbitGrp = fid[orbitPath]
            orbit.loadFromH5(orbitGrp)
    
        return orbit

    @pyre.export
    def getDopplerCentroid(self, frequency):
        import h5py
        import isce3
        import os
        dopplerPath = os.path.join(self.ProcessingInformationPath, 'parameters', frequency, 'dopplerCentroid') 

        with h5py.File(self.filename, 'r') as fid:
            doppler = fid[dopplerPath][:]

        dopplerCentroid = isce3.core.dopplerCentroid(x=self.getSlantRange(), 
                                                    y=self.getZeroDopplerTime(), 
                                                    z=doppler)
        return dopplerCentroid

    @pyre.export
    def getZeroDopplerTime(self):
        import h5py
        import os
        zeroDopplerTimePath = os.path.join(self.ProcessingInformationPath, 
                                          'parameters/zeroDopplerTime')
        with h5py.File(self.filename, 'r') as fid:
            zeroDopplerTime = fid[zeroDopplerTimePath][:]

        return zeroDopplerTime

    @pyre.export
    def getSlantRange(self):
        import h5py
        import os
        slantRangePath = os.path.join(self.ProcessingInformationPath,
                                            'parameters/slantRange')
        with h5py.File(self.filename, 'r') as fid:
            slantRange = fid[slantRangePath][:]

        return slantRange

    @pyre.export
    def getRadarGrid(self, frequency):
        '''
        Returns radarGrid object corresponding to given frequency.
        '''
        raise NotImplementedError

    def parsePolarizations(self):
        '''
        Parse HDF5 and identify polarization channels available for each frequency.
        '''
        import os
        import h5py
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

        with h5py.File(self.filename, 'r') as fid:
            for freq in frequencyList:
                root = os.path.join(folder, 'frequency{0}'.format(freq))
                polList = extractWithIterator(fid[root], 'listOfPolarizations', bytestring,
                                              msg='Could not determine polarization for frequency{0}'.format(freq))
                self.polarizations[freq] = polList

        return

    @property
    def CFPath(self):
        return self._CFPath

    @property
    def RootPath(self):
        return self._RootPath

    @property
    def IdentificationPath(self):
        import os
        return os.path.join(self.RootPath, self._IdentificationPath)

    @property
    def ProductPath(self):
        import os
        return os.path.join(self.RootPath, self.productType)

    @property
    def MetadataPath(self):
        import os
        return os.path.join(self.ProductPath, self._MetadataPath)
    
    @property
    def ProcessingInformationPath(self):
        import os
        return os.path.join(self.MetadataPath, self._ProcessingInformation)

    @property
    def SwathPath(self):
        import os
        return os.path.join(self.ProductPath, self._SwathPath)

    @property
    def GridPath(self):
        import os
        return os.path.join(self.ProductPath, self._GridPath)

    @property
    def productType(self):
        return self.identification.productType

    def populateIdentification(self):
        '''
        Read in the Identification information and assert identity.
        '''
        import h5py
        from .Identification import Identification

        with h5py.File(self.filename, 'r') as fileID:
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

# end of file
