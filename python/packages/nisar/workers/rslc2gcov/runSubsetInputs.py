# -*- coding: utf-8 -*-
import h5py
import collections

def runSubsetInputs(self):
    '''
    This step reads the subset information - freq, pol and spatial.
    '''

    state = self.state
    subset_dict = self.get_value(['processing',
        'input_subset', 'list_of_frequencies'])

    if isinstance(subset_dict, str):
        state.subset_dict = collections.OrderedDict()
        state.subset_dict[subset_dict] = None
    elif not subset_dict:
        state.subset_dict = collections.OrderedDict()
        state.subset_dict['A'] = None
        state.subset_dict['B'] = None
    else:
        state.subset_dict = collections.OrderedDict(subset_dict)

    if 'outputList' not in state.__dir__(): 
        state.outputList = collections.OrderedDict()

    self._radar_grid_list = collections.OrderedDict()
    frequency_list = state.subset_dict.keys()
    for frequency in frequency_list:
        if not state.subset_dict[frequency]:
            current_key = (f'//science/LSAR/SLC/swaths/'
                        f'frequency{frequency}/listOfPolarizations')
            hdf5_obj = h5py.File(state.input_hdf5, 'r')
            if current_key not in hdf5_obj:
                print(f'ERROR key {current_key} not found in'
                      f' {state.input_hdf5}. Ignoring frequency {frequency}.')
                del state.subset_dict[frequency]
                continue
            state.subset_dict[frequency] = [s.decode().upper() 
                for s in hdf5_obj[current_key]]
            hdf5_obj.close()

        # Get radar grid
        radar_grid = self.slc_obj.getRadarGrid(frequency)
        if (state.nlooks_az > 1 or state.nlooks_rg > 1):
            radar_grid_ml = radar_grid.multilook(state.nlooks_az, 
                                                 state.nlooks_rg)
        else:
            radar_grid_ml = radar_grid

        self._radar_grid_list[frequency] = radar_grid_ml

        # Prepare outputList dict
        if 'frequency' not in state.outputList.keys():
            state.outputList[frequency] = []

# end of file
