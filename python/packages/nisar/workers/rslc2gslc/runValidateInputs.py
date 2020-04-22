# -*- coding: utf-8 -*-

'''
This is the first step of the workflow
'''

def runValidateInputs(self, ymlfile, defaults=None, outputState=None):
    '''
    This steps loads the yaml file and sets up initial state of workflow.
    '''

    from . import State

    #Parse yaml
    inputs = self.loadYAML(ymlfile)
    
    ###Create initial state
    state = State()

    ##This section is for logic checks
    ##Update state if needed / flag errors in inputs

    return inputs, state

# end of file
