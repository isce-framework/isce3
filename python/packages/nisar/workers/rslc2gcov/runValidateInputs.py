# -*- coding: utf-8 -*-

'''
This is the first step of the workflow
'''
import os

def runValidateInputs(self):
    '''
    This steps loads the yaml file and sets up initial state
     of workflow.
    '''

    ##This section is for logic checks
    ##Update state if needed / flag errors in inputs

    state = self.state
    state.input_hdf5 = self.get_value(['InputFileGroup', 'InputFilePath'])
    if isinstance(state.input_hdf5, list):
        assert(len(state.input_hdf5) == 1)
        state.input_hdf5 = state.input_hdf5[0]

    if not state.input_hdf5:
        raise AttributeError('ERROR the following argument is required:'
                             'input RSLC')
    
    state.output_hdf5 = self.get_value(['ProductPathGroup', 'SASOutputFile'])
    output_dir = os.path.dirname(state.output_hdf5)
    if output_dir and output_dir != '.':
        os.makedirs(output_dir, exist_ok=True)

    # nlooks
    nlooks_az = self.get_value(['processing', 'pre_process', 'azimuth_looks'])
    nlooks_rg = self.get_value(['processing', 'pre_process', 'range_looks'])
    state.nlooks_az = self.cast_input(nlooks_az, dtype=int, 
        default=1)
    state.nlooks_rg = self.cast_input(nlooks_rg, dtype=int, 
        default=1)

# end of file
