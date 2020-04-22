# -*- coding: utf-8 -*-

'''
This step checks if a DEM needs to be staged or if staged DEM is appropriate. 
'''

def runVerifyDEM(self, config, defaults=None):
    '''
    This step makes sure an appropriate DEM is available for processing.
    '''

    ##This section is for logic checks
    ##Update state if needed / flag errors in inputs
    if (not config['worker']['internet_access']) and (not config['inputs']['dem']):
        raise ValueError('No DEM has been provided and internet_access has been disabled')

    ##This section for exercising DEM access via Virginia's tools
    ##When internet is available and no DEM is explicitly provided


    ##Update self.state as needed

    return

# end of file
