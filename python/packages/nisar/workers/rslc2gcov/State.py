# -*- coding: utf-8 -*-

'''
This file contains the classes relate to storing the state of a workflow.
'''

class State(object):
    '''
    State object for rslc2gcov.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        #Input HDF5 file
        self.input_hdf5 = None

        #Output HDF5 file
        self.output_hdf5 = None

        #Number of looks
        self.nlooks_az = None
        self.nlooks_rg = None

        #Dict of lists - one entry per frequency
        #Represents subset of layers to apply workflow to
        #Auto-detected when not provided by user
        self.subset_dict = None

        #Downloaded dem file
        #Auto-derived when not provided by user
        self.dem_file = None

        #X/Y postings for different frequencies
        #Dict with frequency as keys
        #Auto-derived when not provided by user
        self.geotransform_dict = None

        #Output projection system to use with geotransform
        #Auto-derived when not provided by user
        self.output_epsg = None



# end of file
