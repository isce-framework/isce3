# -*- coding: utf-8 -*-
import h5py
import collections

def runSubsetInputs(self):
    '''
    This step reads the subset information - freq, pol and spatial.
    '''

    state = self.state

    subset_dict = self.get_value(['runconfig', 'groups', 'processing',
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
    for frequency in state.subset_dict.keys():
        if not state.subset_dict[frequency]:
            current_key = (f'//science/LSAR/SLC/swaths/'
                        f'frequency{frequency}/listOfPolarizations')
            hdf5_obj = h5py.File(state.input_hdf5, 'r')
            state.subset_dict[frequency] = [s.decode() 
                for s in hdf5_obj[current_key]]
            hdf5_obj.close()

        # Get radar grid
        radar_grid = self.slc_obj.getRadarGrid(frequency)

        self._radar_grid_list[frequency] = radar_grid

        # Prepare outputList dict
        if 'frequency' not in state.outputList.keys():
            state.outputList[frequency] = []

# end of file
