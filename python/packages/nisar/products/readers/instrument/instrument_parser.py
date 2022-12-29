import h5py
import numpy as np


# Note that this parser is subject to frequent augmentations as new items being
# added to the instrument HDF5 product!


class InstrumentParser:
    """Class for parsing NISAR Instrument HDF5 file.

    The instrument file contains onboard digital beamforming (DBF) tables and
    other instrument-related parameters needed for calibrating or adjusting
    science data for all receive polarization channels.

    Parameters
    ----------
    filename : str
        filename of instrument HDF5 file.

    Attributes
    ----------
    filename : str
        Filename of HDF5 instrument file.
    fid : h5py.File
        File object of HDF5
    pols : list of str
        List of available single-char RX (receive) polarization channel with
        either of 'H' or 'V' values.
    num_channels : int
        Total number of RX channels.
    num_angles_dbf : int
        Number of elevation angles/switches in DBF tables
    num_sets_dbf : int
        Number of sets, a group of datasets, in DBF-related tables
        Usually equal to three, one for each qFSP (
        quadrature first-stage processor)
    version : str
        Version of the product format/spec. The version format is
        {major}.{minor}

    Raises
    ------
    IOError
        If HDF5 file does not exist or can not be opened.
    ValueError
        Missing data.
    KeyError
        Wrong group name or dataset name.

    """

    def __init__(self, filename):
        self._filename = filename
        self._fid = h5py.File(filename, mode='r')

    def __repr__(self):
        return f"{self.__class__.__name__}({self._filename})"

    def __enter__(self):
        return self

    def __exit__(self, val_, type_, tb_):
        self._fid.close()

    @property
    def filename(self):
        return self._filename

    @property
    def fid(self):
        return self._fid

    @property
    def pols(self):
        pols = [item[0]
                for item in self.fid.keys() if item in ['HPOL', 'VPOL']]
        if len(pols) == 0:
            raise KeyError('Groups "HPOL" and "VPOL" are missing!')
        return pols

    @property
    def num_channels(self):
        num_chnl_set = self.fid[
            f'{self.pols[0]}POL/angleToCoefficient/coefSet1'].shape[0]
        return self.num_sets_dbf * num_chnl_set

    @property
    def num_angles_dbf(self):
        return self.fid[
            f'{self.pols[0]}POL/angleToCoefficient/coefSet1'].shape[1]

    @property
    def num_sets_dbf(self):
        list_coefs_set = [key for key in self.fid[
            f'{self.pols[0]}POL/angleToCoefficient'] if 'coef' in key]
        num_sets = len(list_coefs_set)
        if num_sets:
            return num_sets
        raise ValueError('No datasets for DBF coeffs!')

    @property
    def version(self):
        return self.fid['metaData/version'][()].decode()

    def sampling_rate_ta(self, pol):
        """
        Get sampling rate for all Time-to-Angle (TA) datasets of a
        polarization.

        Parameters
        ----------
        pol : {'H', 'V'}
            Rx polarization. Must be a valid polarization in the instrument
            file.

        Returns
        -------
        float
            Sampling rate of switch indexes in (Hz). This is the clock rate
            at which DBF process is performed on instrument!

        Raises
        ------
        ValueError
            For invalid `pol` value.
        """
        if pol not in self.pols:
            raise ValueError(f'Wrong pol! The valid pols are {self.pols}.')
        return self.fid[f'{pol}POL/timeToAngle/samplingRate'][()]

    def get_time2angle(self, pol):
        """Get DBF Time-to-Angle (TA) table for all channels of a polarization

        Parameters
        ----------
        pol : {'H', 'V'}
            Rx polarization. Must be a valid polarization in the instrument
            file.

        Returns
        -------
        np.ndarray(uint32)
            2-D array of fast-time switch indexes @ DBF clock rate where DBF
            EL angle and DBF coeffs changes with shape
            (channels, number of angles or switches)

        Raises
        ------
        ValueError
            For invalid `pol` value.

        """
        if pol not in self.pols:
            raise ValueError(f'Wrong pol! The valid pols are {self.pols}.')
        # number of repeats per set
        n_repeat = self.num_channels // self.num_sets_dbf
        # initialize the TA table
        ta_table = np.zeros((self.num_channels, self.num_angles_dbf),
                            dtype='uint32')
        for n_set in range(self.num_sets_dbf):
            c_slice = slice(n_set * n_repeat,
                            (n_set + 1) * n_repeat)
            ta_table[c_slice] = self.fid[
                f'{pol}POL/timeToAngle/switchIndexSet{n_set + 1}'][()]
        return ta_table

    def el_angles_ac(self, pol):
        """
        Get elevation (EL) angles related to all Angle-to-coefficients (AC)
        datasets of a polarization.

        Parameters
        ----------
        pol : {'H', 'V'}
            Rx polarization. Must be a valid polarization in the instrument
            file.

        Returns
        -------
        np.ndarray(float64)
            2-D array of EL angles in radians with shape
            (channels, number of angles or coeffs).
            These angles are uniformly spaced but can have different angle
            coverage per a set of channels.

        Raises
        ------
        ValueError
            For invalid `pol` value.

        """
        if pol not in self.pols:
            raise ValueError(f'Wrong pol! The valid pols are {self.pols}.')
        # number of channels per set
        n_chl_set = self.num_channels // self.num_sets_dbf
        # initialize the array
        el_ang = np.zeros((self.num_channels, self.num_angles_dbf),
                          dtype='float64')
        for n_set in range(self.num_sets_dbf):
            c_slice = slice(n_set * n_chl_set,
                            (n_set + 1) * n_chl_set)
            el_ang[c_slice] = self.fid[
                f'{pol}POL/angleToCoefficient/angleSet{n_set + 1}'][()]
        return el_ang

    def get_angle2coef(self, pol):
        """
        Get DBF Angle-to-coefficients (AC) table for all channels
        of a polarization.

        Parameters
        ----------
        pol : {'H', 'V'}
            Rx polarization. Must be a valid polarization in the instrument
            file.

        Returns
        -------
        np.ndarray(complex64)
            2-D array of AC coefficients with shape
            (channels, number of angles or coeffs)

        Raises
        ------
        ValueError
            For invalid `pol` value.

        """
        if pol not in self.pols:
            raise ValueError(f'Wrong pol! The valid pols are {self.pols}.')
        # number of channels per set
        n_chl_set = self.num_channels // self.num_sets_dbf
        # initialize the AC table
        ac_table = np.zeros((self.num_channels, self.num_angles_dbf),
                            dtype='complex64')
        for n_set in range(self.num_sets_dbf):
            c_slice = slice(n_set * n_chl_set,
                            (n_set + 1) * n_chl_set)
            ac_table[c_slice] = self.fid[
                f'{pol}POL/angleToCoefficient/coefSet{n_set + 1}'][()]
        return ac_table

    def get_filenames_dbf(self):
        """Get a list of filenames used for generating all DBF tables
        TA and AC.

        This is part of meta data information.

        Returns
        -------
        list of str
            filenames for TA tables
        list of str
            filenames for AC tables

        """
        fn_ta = self.fid['metaData/inputs/timeToAngleFiles'][()].tolist()
        fn_ac = self.fid['metaData/inputs/angleToCoefficientFiles'][()
                                                                    ].tolist()

        return fn_ta, fn_ac
