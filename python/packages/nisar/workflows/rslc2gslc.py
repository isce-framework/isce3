#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-


import os
import argparse
import glob
import time
import datetime 
from collections import defaultdict
import nisar.workers.rslc2gslc as App
from nisar.products.readers import SLC
from ruamel.yaml import YAML
from nisar.workers.rslc2gslc import State

FREQUENCY_LIST = ['A', 'B']
PRINT_CHAR_REPEAT = 70

class Workflow(object):
    '''
    This is the end-to-end workflow for the rslc2gslc NISAR SAS.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        self._slc_obj = None
        self._radar_grid_list = None
        self.verbose = None

        #State object
        self.state = App.State()

        #Saved states folder
        self.state_folder = 'rslc2gslc_completed_steps'
        self.current_state_filename = os.path.join(
            self.state_folder, 'currentState')

        #Steps completed
        self.steps_completed = []

    def save_state(self, step, state_counter):
        '''
        Save state to file.
        '''
        self.steps_completed.append(step)
        db = {}
        db['state'] = self.state.__dict__
        db['steps_completed'] = self.steps_completed
        yaml=YAML(typ='safe')

        step_name = f'{state_counter}_{step}'
        state_filename = os.path.join(self.state_folder, step_name)

        with open(state_filename, 'w') as fid:
            yaml.dump(db, fid)

        if os.path.islink(self.current_state_filename):
            os.unlink(self.current_state_filename)
        if os.path.isfile(self.current_state_filename):
            os.path.remove(self.current_state_filename)
        state_relpath = os.path.relpath(state_filename, 
            os.path.dirname(self.current_state_filename))
        os.symlink(state_relpath, self.current_state_filename)

    def load_state(self, step=None, state_filename=None, step_number=None):
        '''
        Load state from file.
        '''
        if state_filename is None and step_number is not None:
            search_str = os.path.join(self.state_folder, f'{step_number}_*')
            state_filelist = glob.glob(search_str)
            if len(state_filelist) != 1:
                raise AttributeError(f'{self.__class__.__name__}.{__name__}.'
                                f'ERROR reading state from step {step_number}') 
            state_filename = state_filelist[0]
        elif state_filename is None and step is not None:
            state_filename = os.path.join(self.state_folder, step)
        elif state_filename is None:
            raise AttributeError(f'{self.__class__.__name__}.{__name__}.'
                                 'ERROR undefined step')

        self._print(f'loading state from file: {state_filename}') 
        with open(state_filename, 'r') as fid:
            state_filename_str = fid.read()

        yaml = YAML(typ='safe')
        db = yaml.load(state_filename_str)
        state_dict = db['state']

        for key, value in state_dict.items():
            self.state.__setattr__(key, value)

        self.steps_completed = db['steps_completed']
        self._print(f'steps completed: {self.steps_completed}')

    def loadYAML(self, ymlfile):
        '''
        Load yaml and return values.
        '''

        ###Read in the yaml file into inputs
        yaml = YAML(typ='safe')
        with open(ymlfile, 'r') as fid:
            instr = fid.read()

        inputs = yaml.load(instr)

        return inputs


    def run(self, args):
        '''
        Run the workflow with parser arguments.
        '''
        start_time = time.time()

        states_list = ['validateInputs',
                       'verifyDEM',
                       'subsetInputs',
                       'prepHDF5',
                       'geocodeSLC_{frequency}',
                       'finalizeHDF5']

        self.verbose = args.verbose
        self._print('===')
        self._print('RSLC to GSLC (rslc2gslc)')
        self._print('===')
        if args.run_config_filename:
            self._print(f'run_config: {args.run_config_filename}')
        self._print('')
        os.makedirs(self.state_folder, exist_ok=True)

        #Parse yaml
        self.userconfig = self.loadYAML(args.run_config_filename)
        
        ###Create initial state
        if args.resume_from_step is not None:
            self._print(f'resume from step: {args.resume_from_step}')
            self.load_state(step_number=args.resume_from_step-1)
        elif (not args.flag_restart and 
               (os.path.islink(self.current_state_filename) or
                os.path.isfile(self.current_state_filename))):
            self.load_state(state_filename=self.current_state_filename)
        else:
            self.state = State()

        state_counter = 1

        # iterate over states
        for state_str in states_list:

            # worker_name
            worker_name = 'run' + state_str[0].upper()+state_str[1:]

            # calling unparametrized workers
            if '{frequency}' not in state_str:
                state_name = state_str
                self._run_step(state_name, worker_name, state_counter, args)
                state_counter += 1
                continue

            # calling parameter-dependent workers
            worker_name = worker_name.replace('_{frequency}', '')
            for frequency in FREQUENCY_LIST:
                state_name = state_str.replace('{frequency}', frequency)
                worker_kwargs = {}
                worker_kwargs['frequency'] = frequency 
                self._run_step(state_name, worker_name, state_counter, args,
                              worker_kwargs=worker_kwargs)
                state_counter += 1
        self._print('===')
        elapsed_time = time.time() - start_time
        hms_str = str(datetime.timedelta(seconds = int(elapsed_time)))
        self._print(f'elapsed time: {hms_str}s ({elapsed_time:.3f}s)')
        self._print('===')

    def _run_step(self, state_name, worker_name, state_counter, args, 
                  worker_kwargs=None):
        if (not args.flag_restart and 
                state_name in self.steps_completed):
            return
        
        self._print('===')
        self._print(f'step {state_counter}: executing {state_name}')
        self._print('---')
        if args.dry_run:
            return
        worker = getattr(App, worker_name)
        if worker_kwargs is None:
            ret = worker(self)   
        else:
            ret = worker(self, **worker_kwargs)
        if not ret:
            self.save_state(state_name, state_counter)

    # returns values from self.userconfig
    def get_value(self, list_of_keys, default=None):
        input_dict = self.userconfig
        for key in list_of_keys:
            if key not in input_dict.keys():
                return default
            input_dict = input_dict[key]
        if not isinstance(input_dict, dict):
            return input_dict
        ret_dict = defaultdict(lambda: default, input_dict)
        return ret_dict

    def cast_input(self, data_input, dtype=None, default=None,
                   frequency=None):
        if (data_input is None or 
                (isinstance(data_input, str) and data_input.lower() == 'none')):
            return default
        if (frequency is not None and
                not isinstance(data_input, str) and 
                hasattr(data_input, '__getitem__')):
            index = FREQUENCY_LIST.index(frequency)
            data_input =  data_input[index]
        if dtype is None:
            return data_input
        return dtype(data_input)


    @property
    def slc_obj(self):
        if self._slc_obj is None:
            self._slc_obj = SLC(hdf5file=self.state.input_hdf5)
        return self._slc_obj

    @property
    def orbit(self):
        return self.slc_obj.getOrbit()

    @property
    def doppler(self):
        tempSlc = SLC(hdf5file=self.state.input_hdf5)
        dop = tempSlc.getDopplerCentroid()
        
        return dop

    @property
    def radar_grid_list(self):
        '''Dict of radargrids - one entry per frequency
        Stores radar grid parameters corresponding to each frequency
        '''
        if self._radar_grid_list is None:
            App.runSubsetInputs(self)
        return self._radar_grid_list

    def _print(self, message, *args, **kwargs):
        if not self.verbose:
            return
        if message == '===':
            print(PRINT_CHAR_REPEAT*'=', *args, **kwargs)
            return
        if message == '---':
            print(PRINT_CHAR_REPEAT*'-', *args, **kwargs)
            return
        print(message, *args, **kwargs)


def cmdLineParse():
    '''
    Command line parser for rslc2gslc
    '''

    parser = argparse.ArgumentParser(description='rslc2gslc.py - Generate GSLC'
        ' from RSLC product', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('run_config_filename', 
                        type=str,
                        help='Input run config file')

    step_options = parser.add_mutually_exclusive_group() 

    step_options.add_argument('--resume-from-step', 
                              '--jump-to-step', 
                              dest='resume_from_step', 
                              type=int,
                              required=False,
                              help='Resume from specific step')

    step_options.add_argument('--restart',
                              '--start',
                              '--force-restart',
                              dest='flag_restart',
                              action='store_true',
                              help='Restart processing',
                              default=False)
    step_options.add_argument('--continue',
                                '--force-continue',
                                '--resume',
                                '--force-resume',
                                dest='flag_restart',
                                action='store_false',
                                help='Continue from last step',
                                default=False)

    parser.add_argument('--dry-run', 
                        dest='dry_run', 
                        action='store_true',
                        help='Dry run')

    parser_verbose = parser.add_mutually_exclusive_group()
    parser_verbose.add_argument('-q',
                                '--quiet',
                                dest='verbose',
                                action='store_false',
                                help='Activate quiet (non-verbose) mode',
                                default=True)
    parser_verbose.add_argument('-v',
                                '--verbose',
                                dest='verbose',
                                action='store_true',
                                help='Activate verbose mode',
                                default=True)
    
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
