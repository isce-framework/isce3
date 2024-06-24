# -*- coding: utf-8 -*-
import h5py
import journal
import numpy as np

from nisar.products.readers.Base import Base, get_hdf5_file_root_path


def get_hdf5_file_product_type(filename: str, root_path: str = None) -> str:
    """
    Return product type from NISAR product (HDF5 file).

    Parameters
    ----------
    filename : str
        HDF5 filename
    root_path : str (optional)
        Preliminary root path to check before default root
        path list. This option is intended for non-standard NISAR products.

    Returns
    -------
    str
        Product type
    """
    if root_path is None:
        root_path = get_hdf5_file_root_path(filename, root_path=root_path)

    with h5py.File(filename, 'r', libver='latest', swmr=True) as f:
        product_type_ds = f[root_path+'/identification/productType']
        product_type = str(np.asarray(product_type_ds, dtype=str))

        # The product group name should be "RSLC" per the spec.
        # However, early sample products used "SLC" instead.
        # We maintain compatibility with both options.

        if product_type == 'SLC':
            return 'RSLC'
        return product_type


class GenericProduct(Base, family='nisar.productreader.product'):
    """
    Class for parsing NISAR products into isce3 structures.
    """

    def __init__(self, **kwds):
        """
        Constructor to initialize product with HDF5 file.
        """

        # Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('GenericProduct')

        self.identification.productType = \
            get_hdf5_file_product_type(self.filename,
                                       root_path = self.RootPath)

        self.parsePolarizations()

    def parsePolarizations(self):
        """
        Parse HDF5 and identify polarization channels available for each frequency.
        """
        from nisar.h5 import bytestring, extractWithIterator

        try:
            frequencyList = self.frequencies
        except:
            self.error_channel.log(
                'Cannot determine list of available frequencies'
                ' without parsing Product Identification')

        ###Determine if product has swaths / grids

        folder_list = [self.SwathPath, self.GridPath]

        flag_found_folder = False
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for folder in folder_list:
                for freq in frequencyList:
                    root = f"{folder}/frequency{freq}"
                    if root not in fid:
                        continue
                    flag_found_folder = True
                    polList = extractWithIterator(
                        fid[root], 'listOfPolarizations', bytestring,
                        msg=f"Could not determine polarization for frequency{freq}",
                    )
                    self.polarizations[freq] = [p.upper() for p in polList]
                if flag_found_folder:
                    break

    def getProductLevel(self):
        """
        Returns the product level
        """
        if self.productType in ['GCOV', 'GSLC', 'GUNW', 'GOFF']:
            return "L2"
        if self.productType in ['RSLC', 'RIFG', 'RUNW', 'ROFF']:
            return "L1"
        if self.productType in ['RRSD']:
            return "L0B"
        return "undefined"
    
    def getImageDataset(
        self,
        frequency: str,
        polarization: str,
        **kwargs
    ) -> h5py.Dataset:
        """
        Returns the primary image dataset for the given frequency, polarization, etc.

        Parameters
        ----------
        frequency: "A" or "B"
            The frequency letter associated with the data array.
        polarization: str
            The polarization or covariance term associated with the data array.
            Generally one of "HH", "HV", "VH", "VV".
            For GCOV, may be e.g. "HHHH", "HVHH", "VHVH", etc.
        **kwargs : dict, optional
            Additional product-specific arguments, e.g. layer number for ROFF/GOFF
            products.
        
        Returns
        -------
        h5py.Dataset
            The primary array of data associated with this product at the given
            frequency and polarization, in its native format.
        """
        # open H5 with swmr mode enabled
        fid = h5py.File(self.filename, 'r', libver='latest', swmr=True)

        # build path the desired dataset
        ds_path = self.imageDatasetPath(frequency, polarization, **kwargs)

        # get and return dataset
        return fid[ds_path]
    
    def imageDatasetPath(
        self,
        frequency: str,
        polarization: str,
        **kwargs,
    ) -> str:
        """
        Returns the primary image dataset path for the given frequency, polarization,
        etc.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter (either "A" or "B").
        polarization : str
            A polarization or covariance term to get parameters for.
            For GSLC products, use a polarization. (e.g. "HH", "VV", "HV", etc.)
            For GCOV products, use a covariance term. (e.g. "HHHH", "HVHV", etc.)
        **kwargs : dict, optional
            Additional product-specific arguments, e.g. layer number for ROFF/GOFF
            products.
        
        Returns
        -------
        str
            The path to the primary array of data associated with this product at the
            given frequency and polarization.
        """
        raise NotImplementedError(
            "GenericProduct cannot be used to acquire a product data path. "
            "Please use a subclass (e.g. GSLC, RSLC) to use this functionality."
        )
