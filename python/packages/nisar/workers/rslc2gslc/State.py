# -*- coding: utf-8 -*-

'''
This file contains the classes relate to storing the state of a workflow.
'''

class State(object):
    '''
    State object for rslc2gslc.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        #Input HDF5 file
        self.inputHDF5 = None

        #Output HDF5 file
        self.outputHDF5 = None

        #Dict of lists - one entry per frequency
        #Represents subset of layers to apply workflow to
        #Auto-detected when not provided by user
        self.subsetList = None

        #Dict of radargrids - one entry per frequency
        #Stores radar grid parameters corresponding to each frequency
        self.radarGridList = None

        #Downloaded dem file
        #Auto-derived when not provided by user
        self.downloadDEM = None

        #X/Y postings for different frequencies
        #Dict with frequency as keys
        #Auto-derived when not provided by user
        self.geoTransformList = None

        #Output projection system to use with geotransform
        #Auto-derived when not provided by user
        self.outputEPSG = None

    @staticmethod
    def loadFromFile(infile):
        '''
        Load from picklefile
        '''
        import pickle
        with open(infile, 'rb') as fid:
            data = pickle.load(fid)

        return data

    def saveToFile(self, infile):
        '''
        Write to picklefile
        '''
        import pickle
        with open(inffile, 'wb') as fid:
            pickle.dump(self, fid)
        
        return

# end of file
