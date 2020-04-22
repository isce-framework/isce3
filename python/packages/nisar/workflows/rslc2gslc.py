#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-


import nisar.workers.rslc2gslc as App


class Workflow(object):
    '''
    This is the end-to-end workflow for the rslc2gslc NISAR SAS.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        #State object
        self.state = App.State()

        #Saved states folder
        self.state_folder = 'PICKLE'

        #Steps completed
        self.steps_completed = []

    def saveState(self, step=None):
        '''
        Save state to pickle file.
        '''
        import shelve
        import os

        self.steps_completed.append(step)
        with shelve.open( os.path.join(self.state_folder, step), 'n') as db:
            db['state'] = self.state
            db['steps_completed'] = self.steps_completed

    def loadState(self, step=None):
        '''
        Load state from pickle file.
        '''
        import shelve
        import os

        with shelve.open( os.path.join(self.state_folder, step), 'r') as db:
            self.state = db['state']
            self.steps_completed = db['steps_completed']

    def loadYAML(self, ymlfile):
        '''
        Load yaml and return values.
        '''

        from ruamel.yaml import YAML

        ###Read in the yaml file into inputs
        yaml = YAML()
        with open(ymlfile, 'r') as fid:
            instr = fid.read()

        inputs = yaml.load(instr)

        return inputs


    def run(self, args):
        '''
        Run the workflow with parser arguments.
        '''
        import os

        os.makedirs(self.state_folder, exist_ok=True)
 
        ##First step
        userconfig, state = App.runValidateInputs(self, args.yml)
        self.state = state
        self.saveState('validateInputs')

        ##Second step
        self.loadState('validateInputs')
        App.runVerifyDEM(self, userconfig)
        self.saveState('verifyDEM')

        ##Third step
        self.loadState('verifyDEM')
        App.runSubsetInputs(self, userconfig)
        self.saveState('subsetInputs')

        ##Fourth step
        self.loadState('subsetInputs')
        App.runPrepHDF5(self, userconfig)
        self.saveState('prepHDF5')

        ##Fifth step
        self.loadState('prepHDF5')
        App.runGeo2rdr(self, userconfig)
        self.saveState('geo2rdr')

        ##Sixth step
        self.loadState('geo2rdr')
        App.runResampleSLC(self, userconfig)
        self.saveState('resampleSLC')

        ##Seventh step
        self.loadState('resampleSLC')
        App.runFinalizeHDF5(self, userconfig)
        self.saveState('finalizeHDF5')


def cmdLineParse():
    '''
    Command line parser for rslc2gslc
    '''
    import argparse

    parser = argparse.ArgumentParser(description='rslc2gslc.py - Generate GSLC from RSLC product')
    parser.add_argument('-r', dest='yml', type=str,
                              required=True,
                              help='Input run config file')

    return parser.parse_args()



if __name__ == '__main__':
    '''
    Main driver.
    '''

    #Parse command line arguments
    inps = cmdLineParse()


    #Create state variable for workflow
    workflow = Workflow()

    #Execute workflow
    workflow.run(inps)

# end of file
