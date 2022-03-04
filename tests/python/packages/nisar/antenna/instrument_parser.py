import h5py
import numpy as np

class InstrumentParser:
    """Class for parsing NISAR Instrument table HDF5 file.

    Attributes
    ----------
    file_name: str
        file name of instrument table HDF5
    h5: h5py.File
        File object of instrument table HDF5
    qfsps: constant array of str
        Quad-First Stage Processor (qFSP) FPGA numbers

    """

    # Number of beams/channels per qFSP
    NUM_CHAN_QFSP = 4

    def __init__(self, file_name, qfsp=None):
        self._file_name = file_name
        self._h5 = h5py.File(file_name, mode="r", libver="latest", swmr=True)
        self._qfsps = qfsp if qfsp is not None else ["0", "1", "2"]

    @property
    def file_name(self):
        return self._file_name

    @property
    def h5(self):
        return self._h5

    @property
    def qfsps(self):
        return self._qfsps

    def _check_pol(self, pol: str) -> str:
        """Check and get upper-case polarization type."""
        pol = pol.upper()
        if pol not in ["H", "V"]:
            raise ValueError("'pol' shall be either 'H' or 'V'")
        return pol

    def ac_path(self):
        """Return Angle-to-Coefficient (AC) Table Path."""
        return "/instrumentTables/ACTables"

    def ta_path(self):
        """Return Time-To-Angle (TA) Table Path."""
        return "/instrumentTables/TATables"

    def get_ac_angle_count(self, pol="H"):
        """Return Angle-To-Coefficient (A)C Table angle count.
           This value is 256 for NISAR.

        Parameters
        ----------
        pol: str, optional
            Rx polarization of the beam, either
            `H` or `V'. It is case insensitive. (default: 'H')

        Returns
        -------
        ac_angle_count: int
            number of coefficient in the AC table, NISAR = 256.
        """
        pol = self._check_pol(pol)

        ac_angle_count_path = f"{self.ac_path()}/qFSP{pol}/{pol}0/angleCount"
        ac_angle_count = int(np.asarray(self._h5[ac_angle_count_path]))

        return ac_angle_count

    def get_ac_first_last_angles(self, pol="H"):
        """Extract Angle-to-Coefficient(AC) Table first and last angles
           NISAR: ~ -7 - 4 deg

        Parameters
        ----------
            pol: str, default='H'
                Polarization of the beam , either `H` or `V'. It is case insensitive.

        Returns
        -------
            ac_first_angle: array of float
                Elevation angle, in degrees with respect to the boresight angle, corresponding to
                the first weighting coefficient in the AC table with respect to each qFSP.
                one value per QFSP, dim = 3 x 1. 
            ac_last_angle: array of float
                Elevation angle, in degrees with respect to the boresight angle, corresponding to
                the last weighting coefficient in the AC table with respect to each qFSP. 
                one value per QFSP, dim = 3 x 1. 
        """
        pol = self._check_pol(pol)

        num_qfsp = len(self._qfsps)
        ac_first_angle = np.zeros(num_qfsp)
        ac_last_angle = np.zeros(num_qfsp)

        for idx, qfsp in enumerate(self._qfsps):
            ac_first_angle_path = f"{self.ac_path()}/qFSP{pol}/{pol}{qfsp}/firstAngle"
            ac_last_angle_path = f"{self.ac_path()}/qFSP{pol}/{pol}{qfsp}/lastAngle"

            first_angle = np.asarray(self._h5[ac_first_angle_path])
            last_angle = np.asarray(self._h5[ac_last_angle_path])

            ac_first_angle[idx] = first_angle.item()
            ac_last_angle[idx] = last_angle.item()

        return ac_first_angle, ac_last_angle

    def get_ac_angles_low_high(self, pol="H"):
        """Extract Angle-to-Coefficient (AC) Table channel low and high angle indices
        The low angle index is the index of the smallest angle for which the antenna gain of the
        local channel is significant for the beamforming process. The high angle index
        is the index of largest angle for which this is true.

        Parameters:
        ----------
            pol: str, default='H'
                Polarization of the beam , either
                `H` or `V'. It is case insensitive.

        Returns:
        -------
            angle_low_idx: array of int
                Lowest angle index (0 to 255) where the antenna gain of a Local Channel
                is significant for the beamforming process.
                one index per channel, dim = [num of channels]
            angle_hight_idx: array of int
                Highest angle index (0 to 255) where the antenna gain of a Local Channel
                is significant for the beamforming process.
                one index per channel, dim = [num of channels]
        """

        pol = self._check_pol(pol)

        num_qfsp = len(self._qfsps)
        chan_idx = np.arange(self.NUM_CHAN_QFSP)
        angle_low_idx = np.zeros(num_qfsp * self.NUM_CHAN_QFSP, dtype=int)
        angle_high_idx = np.zeros(num_qfsp * self.NUM_CHAN_QFSP, dtype=int)

        for idx, qfsp in enumerate(self._qfsps):
            angle_low_path = f"{self.ac_path()}/qFSP{pol}/{pol}{qfsp}/alow"
            angle_high_path = f"{self.ac_path()}/qFSP{pol}/{pol}{qfsp}/ahigh"

            angle_low_idx[chan_idx] = self._h5[angle_low_path][:]
            angle_high_idx[chan_idx] = self._h5[angle_high_path][:]

            chan_idx += self.NUM_CHAN_QFSP

        return angle_low_idx, angle_high_idx

    def get_ac_coef(self, pol="H"):
        """Extract Angle-to-Coefficient (AC) Table DBF coefficient values

        Parameters:
        ----------
            pol: str, default='H'
                Polarization of the beam , either
                `H` or `V'. It is case insensitive.

        Returns:
        -------
            ac_chan_coef: 2-D array of complex
                AC Table DBF channel coefficients, dim = [num of chan  x num of coefficients]
        """

        pol = self._check_pol(pol)

        num_coef = self.get_ac_angle_count(pol)
        num_qfsp = len(self._qfsps)
        chan_idx = np.arange(self.NUM_CHAN_QFSP)
        ac_coef_real = np.zeros((num_qfsp * self.NUM_CHAN_QFSP, num_coef))
        ac_coef_imag = np.zeros((num_qfsp * self.NUM_CHAN_QFSP, num_coef))

        for idx, qfsp in enumerate(self._qfsps):
            ac_coef_real_path = f"{self.ac_path()}/qFSP{pol}/{pol}{qfsp}/coeffReal"
            ac_coef_imag_path = f"{self.ac_path()}/qFSP{pol}/{pol}{qfsp}/coeffImag"

            ac_coef_real[chan_idx] = self._h5[ac_coef_real_path][:].transpose()
            ac_coef_imag[chan_idx] = self._h5[ac_coef_imag_path][:].transpose()

            ac_coef = ac_coef_real + 1j * ac_coef_imag

            chan_idx += self.NUM_CHAN_QFSP

        return ac_coef

    def get_ta_dbf_switch(self, pol="H"):
        """Extract Time-To-Angle (TA) Table DBF_SWITCH values.

        Parameters
        ----------
            pol: str, default='H'
                Polarization of the beam , either
                `H` or `V'. It is case insensitive.
        Returns
        -------
            ta_dbf_switch: 2-D array of int
                ta_dbf_switch represents the time to apply individual DBF coefficients stored 
                in the AC table. The number stored in each of DBF_SWITCH is the number of 
                96-MHz clock cycles divided by 2 after RD clock cycles have elapses after 
                the PRF trigger event.
        """

        pol = self._check_pol(pol)

        num_coef = self.get_ac_angle_count(pol)
        num_qfsp = len(self._qfsps)
        ta_dbf_switch = np.zeros((num_qfsp, num_coef))

        # Parse Time to Angle LUT
        for idx, qfsp in enumerate(self._qfsps):
            ta_dbf_switch_path = f"{self.ta_path()}/qFSP{pol}/{pol}{qfsp}/dbf_switch"
            ta_dbf_switch[idx] = self._h5[ta_dbf_switch_path][:]

        return ta_dbf_switch
