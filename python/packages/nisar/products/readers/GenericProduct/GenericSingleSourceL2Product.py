# -*- coding: utf-8 -*-
from __future__ import annotations

import h5py
import journal
import pyre
import numpy as np

import isce3
from isce3.core import DateTime, LookSide, speed_of_light
from isce3.product import GeoGridParameters, RadarGridParameters
from nisar.products.readers.GenericProduct import (
    GenericProduct,
    get_hdf5_file_product_type,
)


class GenericSingleSourceL2Product(
    GenericProduct,
    family='nisar.productreader.product',
):
    """
    Class for parsing NISAR L2 products into isce3 structures.
    """

    def __init__(self, **kwds):
        """
        Constructor to initialize product with HDF5 file.
        """

        ###Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('GenericSingleSourceL2Product')

        self.identification.productType = \
            get_hdf5_file_product_type(self.filename,
                                       root_path = self.RootPath)

        SINGLE_SOURCE_L2_PRODUCT_LIST = ['GCOV', 'GSLC']

        if self.identification.productType not in SINGLE_SOURCE_L2_PRODUCT_LIST:
            error_msg = (
                f"Input HDF5 file {self.filename} is not a valid NISAR "
                "single-source L2 product."
            )
            self.error_channel.log(error_msg)
            raise RuntimeError(error_msg)

        self.parsePolarizations()

    def parsePolarizations(self):
        """
        Parse HDF5 and identify polarization channels available for each frequency.
        """
        from nisar.h5 import bytestring, extractWithIterator

        try:
            frequencyList = self.frequencies
        except:

            error_msg = ('Cannot determine list of available frequencies'
                         ' without parsing Product Identification')
            self.error_channel.log(error_msg)
            raise RuntimeError(error_msg)

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for freq in frequencyList:
                root = f"{self.GridPath}/frequency{freq}"
                if root not in fid:
                    continue
                polList = extractWithIterator(
                    fid[root], 'listOfPolarizations', bytestring,
                    msg=f'Could not determine polarization for frequency{freq}')
                self.polarizations[freq] = [p.upper() for p in polList]

    def getGeoGridProduct(self):
        """
        Returns the GeoGridProduct object for the product.
        """
        return isce3.product.GeoGridProduct(self.filename)

    def getProductLevel(self):
        """
        Returns the product level.
        """
        return "L2"

    @pyre.export
    def getDopplerCentroid(
        self,
        frequency: str | None = None,
    ) -> isce3.core.LUT2d:
        """
        Returns the Doppler centroid for the given frequency.

        Parameters
        ----------
        frequency : "A" or "B" or None, optional
            The frequency letter (either "A" or "B") or None. Returns the LUT of the
            first frequency on the product if None. Defaults to None.

        Returns
        -------
        isce3.core.LUT2d
            The Doppler centroid LUT
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        doppler_group_path = (
            f'{self.sourceDataProcessingInfoPath}/parameters/frequency{frequency}'
        )

        doppler_dataset_path = f'{doppler_group_path}/dopplerCentroid'
        zero_doppler_time_dataset_path = (f'{doppler_group_path}/'
                                          'zeroDopplerTime')
        slant_range_dataset_path = f'{doppler_group_path}/slantRange'

        # extract the native Doppler dataset
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:

            doppler = fid[doppler_dataset_path][:]
            zeroDopplerTime = fid[zero_doppler_time_dataset_path][:]
            slantRange = fid[slant_range_dataset_path][:]

        dopplerCentroid = isce3.core.LUT2d(xcoord=slantRange,
                                           ycoord=zeroDopplerTime,
                                           data=doppler)

        return dopplerCentroid

    def imageDatasetPath(
        self,
        frequency: str,
        polarization: str
    ) -> str:
        # implementation of GenericProduct.imageDatasetPath
        return f"{self.GridPath}/frequency{frequency}/{polarization}"

    @property
    def sourceDataPath(self) -> str:
        return self.MetadataPath + "/sourceData"

    @property
    def sourceDataSwathsPath(self) -> str:
        return self.sourceDataPath + "/swaths"

    @property
    def sourceDataProcessingInfoPath(self) -> str:
        return self.sourceDataPath + "/processingInformation"

    def centerFrequencyPath(self, frequency: str) -> str:
        """
        Return the path to the center frequency dataset.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter (either "A" or "B").

        Returns
        -------
        str
            The path to the center frequency dataset of this frequency on the product.
        """
        return (f"{self.sourceDataSwathsPath}/frequency{frequency}/"
                "centerFrequency")

    def getCenterFrequency(self, frequency: str) -> np.float64:
        """
        Return the value at the center frequency dataset.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter (either "A" or "B").

        Returns
        -------
        np.float64
            The center frequency of this frequency on the product, in Hertz.
        """
        dataset_path = self.centerFrequencyPath(frequency=frequency)

        # open H5 with swmr mode enabled
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            # get dataset
            frequency_dataset: h5py.Dataset = fid[dataset_path]
            frequency_val = frequency_dataset[()]

        return frequency_val

    def getWavelength(self, frequency: str) -> np.float64:
        """
        Return the center wavelength.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter (either "A" or "B").

        Returns
        -------
        np.float64
            The wavelength of this frequency on the product, in meters.
        """
        freq = self.getCenterFrequency(frequency=frequency)
        return speed_of_light / freq

    def getSourceRadarGridParameters(
        self,
        frequency: str | None = None,
    ) -> RadarGridParameters:
        """
        Create a RadarGridParameters from the source data for this product.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        isce3.product.RadarGridParameters
            A RadarGridParameters object representing the properties of the source
            data radar grid.
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        if frequency not in {"A", "B"}:
            raise ValueError("Frequency must be 'A' or 'B'.")

        src_swaths_path = self.sourceDataSwathsPath
        src_freq_path = f"{src_swaths_path}/frequency{frequency}"

        lookside_path = f"{self.sourceDataPath}/lookDirection"
        az_start_path = f"{src_swaths_path}/zeroDopplerStartTime"
        # Wavelength is calculated from center frequency.
        center_freq_path = f"{src_freq_path}/centerFrequency"
        # Pulse Repetition Frequency is the reciprocal of zero doppler time spacing.
        time_spacing_path = f"{src_swaths_path}/zeroDopplerTimeSpacing"
        rg_start_path = f"{src_freq_path}/slantRangeStart"
        range_spacing_path = f"{src_freq_path}/slantRangeSpacing"
        az_length_path = f"{src_swaths_path}/numberOfAzimuthLines"
        rg_width_path = f"{src_freq_path}/numberOfRangeSamples"

        # extract the datasets needed for creating a RadarGridParameters object
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            lookside_str = fid[lookside_path][()].decode()
            sensing_start = fid[az_start_path][()].decode()
            center_freq = fid[center_freq_path][()]
            wavelength = speed_of_light / center_freq
            time_spacing = fid[time_spacing_path][()]
            prf = 1 / time_spacing
            range_start = fid[rg_start_path][()]
            range_spacing = fid[range_spacing_path][()]
            az_length = fid[az_length_path][()]
            rg_width = fid[rg_width_path][()]

        lookside = LookSide.Right if lookside_str.title() == "Right" else LookSide.Left
        ref_epoch = self.getOrbit().reference_epoch
        sensing_start_delta = DateTime(sensing_start) - ref_epoch
        
        return RadarGridParameters(
            sensing_start=sensing_start_delta.total_seconds(),
            wavelength=wavelength,
            prf=prf,
            starting_range=range_start,
            range_pixel_spacing=range_spacing,
            lookside=lookside,
            length=az_length,
            width=rg_width,
            ref_epoch=ref_epoch,
        )

    def projectionPath(self, frequency: str | None = None) -> str:
        """
        Get the path to the HDF5 projection dataset for the given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        dataset_path : str
            The path to the dataset containing the projection on the product.
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        return f"{self.GridPath}/frequency{frequency}/projection"

    def getProjectionEpsg(self, frequency: str | None = None) -> np.int32:
        """
        Return the HDF5 EPSG projection dataset of the given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        epsg: numpy.int32
            The EPSG value of the product.
        """
        # open H5 with swmr mode enabled
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:

            # build path the desired dataset
            dataset_path = self.projectionPath(frequency=frequency)

            # get dataset
            dataset: h5py.Dataset = fid[dataset_path]

            epsg = dataset[()]

        # return EPSG projection
        return epsg

    def xCoordinatesPath(self, frequency: str | None = None) -> str:
        """
        Return path to HDF5 xCoordinates dataset of given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        dataset_path : str
            The path to the dataset containing the X coordinates on the product.
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        return f"{self.GridPath}/frequency{frequency}/xCoordinates"

    def xCoordinateSpacingPath(self, frequency: str | None = None) -> str:
        """
        Return path to HDF5 xCoordinates spacing dataset of given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        dataset_path : str
            The path to the dataset containing the X spacing value on the product.
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        return f"{self.GridPath}/frequency{frequency}/xCoordinateSpacing"

    def yCoordinatesPath(self, frequency: str | None = None) -> str:
        """
        Return path to HDF5 yCoordinates dataset of given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        dataset_path : str
            The path to the dataset containing the Y coordinates on the product.
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        return f"{self.GridPath}/frequency{frequency}/yCoordinates"

    def yCoordinateSpacingPath(self, frequency: str | None = None) -> str:
        """
        Return path to HDF5 yCoordinates spacing dataset of given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        dataset_path : str
            The path to the dataset containing the Y spacing value on the product.
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        return f"{self.GridPath}/frequency{frequency}/yCoordinateSpacing"
    
    def getGeoGridCoordinateDatasets(
        self,
        frequency: str | None = None,
    ) -> tuple[h5py.Dataset, h5py.Dataset]:
        """
        Return HDF5 coordinates datasets of given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        x_dataset, y_dataset : h5py.Dataset
            The datasets containing the X and Y coordinates of the image grid.
        """

        # build path the desired datasets
        x_dataset_path = self.xCoordinatesPath(frequency=frequency)
        y_dataset_path = self.yCoordinatesPath(frequency=frequency)

        # open H5 with swmr mode enabled
        fid = h5py.File(self.filename, 'r', libver='latest', swmr=True)

        # get datasets
        x_dataset: h5py.Dataset = fid[x_dataset_path]
        y_dataset: h5py.Dataset = fid[y_dataset_path]

        # return dataset
        return (x_dataset, y_dataset)
    
    def getGeoGridCoordinateSpacing(
        self,
        frequency: str | None = None,
    ) -> tuple[np.float64, np.float64]:
        """
        Return coordinate spacing values of the given frequency.

        Parameters
        ----------
        frequency : "A" or "B", or None
            The frequency letter (either "A" or "B"). If None, will use A if present
            on the product, or B otherwise. Defaults to None.

        Returns
        -------
        x_spacing : np.float64
            The X-coordinate spacing, in the units of the parent product.
        y_spacing : np.float64
            The Y-coordinate spacing, in the units of the parent product.
        """
        # open H5 with swmr mode enabled
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:

            # build path the desired datasets
            x_dataset_path = self.xCoordinateSpacingPath(frequency=frequency)
            y_dataset_path = self.yCoordinateSpacingPath(frequency=frequency)

            # get datasets
            x_dataset: h5py.Dataset = fid[x_dataset_path]
            y_dataset: h5py.Dataset = fid[y_dataset_path]

            x_spacing = x_dataset[()]
            y_spacing = y_dataset[()]

        # return dataset
        return (x_spacing, y_spacing)
    
    def getGeoGridParameters(
        self,
        frequency: str,
        polarization: str,
    ) -> GeoGridParameters:
        """
        Create a GeoGridParameters from the source data for this product.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter (either "A" or "B").
        polarization : str
            A polarization or covariance term to get parameters for.
            For GSLC products, use a polarization. (e.g. "HH", "VV", "HV", etc.)
            For GCOV products, use a covariance term. (e.g. "HHHH", "HVHV", etc.)

        Returns
        -------
        isce3.product.GeoGridParameters
            The generated GeoGridParameters object.
        """
        x_coords, y_coords = self.getGeoGridCoordinateDatasets(frequency=frequency)
        x_spacing, y_spacing = self.getGeoGridCoordinateSpacing(frequency=frequency)

        length, width = self.getImageDataset(
            frequency=frequency,
            polarization=polarization
        ).shape

        return GeoGridParameters(
            start_x=x_coords[0] - x_spacing / 2,
            start_y=y_coords[0] - y_spacing / 2,
            spacing_x=x_spacing,
            spacing_y=y_spacing,
            width=int(width),
            length=int(length),
            epsg=self.getProjectionEpsg(frequency=frequency)
        )
