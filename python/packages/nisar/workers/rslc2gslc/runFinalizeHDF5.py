# -*- coding: utf-8 -*-

def runFinalizeHDF5(self, defaults=None):
    '''
    This step fills out the HDF5 as needed
    '''
    state = self.state
    for frequency in state.subset_dict.keys():
        self._print(f'updated HDF5 datasets (freq. {frequency}):')
        for h5_ref in state.outputList[frequency]:
            self._print(f'    {h5_ref}')

# end of file
