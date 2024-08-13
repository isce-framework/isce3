from __future__ import annotations

import h5py

from .. import GenericProduct


class SLCBase(GenericProduct, family='nisar.productreader.slcbase'):
    def getSlcDataset(self, frequency: str, polarization: str) -> h5py.Dataset:
        '''
        Return SLC dataset of given frequency and polarization from HDF5 file.

        Note: Currently a simpler reimplementation of self.getImageDataset.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter (either "A" or "B").
        polarization : str
            The Tx and Rx polarization (e.g. "HH", "VV", "HV", etc.).
        
        Returns
        -------
        slc_dataset : h5py.Dataset
            The SLC dataset on the product, for the given frequency and polarization.
        '''
        return self.getImageDataset(frequency=frequency, polarization=polarization)
    
    def slcPath(self, frequency: str, polarization: str) -> str:
        '''
        Return path to HDF5 SLC dataset of given frequency and polarization

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter (either "A" or "B").
        polarization : str
            The Tx and Rx polarization (e.g. "HH", "VV", "HV", etc.).
        
        Returns
        -------
        slc_datset_path : str
            The path to the SLC dataset on the product, for the given frequency and
            polarization.
        '''
        return self.imageDatasetPath(frequency=frequency, polarization=polarization)
