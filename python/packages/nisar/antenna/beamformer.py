import numpy as np
import numpy.matlib
import warnings
import isce3
from isce3 import antenna as ant
from isce3.core import Ellipsoid, Quaternion
from isce3.geometry import DEMInterpolator as DEMInterp
from nisar.antenna import CalPath
from enum import IntEnum, unique


class ElevationBeamformer:
    """Beamformer class that computes Tx & Rx beamformed gain patterns.

    Attributes:
    -----------
    orbit: isce3.core.Orbit
        ISCE3 Orbit object
    attitude: isce3.core.Attitude
        ISCE3 Attitude object
    dem: isce3.geometry.DEMInterpolator
        ISCE3 DEMInterpolator object
    slant_range: isce3.core.Linspace
        ISCE3 SlantRange object
    el_ant_info: namedtuple
        Info about antenna elevation-cut complex multi-channel patterns
        ant_gain_el: np.ndarray(complex)
            2-D complex patterns of shape num_chan (beams) x (number of el angles)
        el_angles_rad : np.ndarray(float)
            array of elevation angles in (rad)
        az_cut_angles_rad : float
            AZ angle at which EL-cut patterns are provided in (rad)
    tx_trm_info: nisar.antenna.TxTrmInfo
        data class that contains all relevant attributes needed to compute Tx beamformed gain pattern
    rx_trm_info: nisar.antenna.RxTrmInfo
        data class that contains all relevant attributes needed to compute Rx beamformed gain pattern
    tx_weight_norm: 2D array of complex
        power normalized unity-gain Tx beamforming weight.
        dim = [num of channel x num of HCAL pulses]
    rx_weight_norm: 2D array of complex
        power normalized unity-gain rx beamforming weight.
        Power normalization is done prior to writing Angle-to-Coefficient LUT to HD5 file.
        dim = [num of channel x num of entries in Angle to Coefficient LUT]
    """

    def __init__(
        self, orbit, attitude, dem, slant_range, el_ant_info, tx_trm_info, rx_trm_info
    ):
        self.orbit = orbit
        self.attitude = attitude
        self.dem = dem
        self.slant_range = slant_range
        self.el_ant_info = el_ant_info
        self.tx_trm_info = tx_trm_info
        self.rx_trm_info = rx_trm_info
        self._tx_weight_norm = self._compute_transmit_pattern_weights()
        self._rx_weight_norm, self._data_fs = self._compute_receive_pattern_weights()

        if self.attitude.reference_epoch != self.orbit.reference_epoch:
            raise ValueError(
                "Orbit and attitude data must have the same reference epoch."
            )

    def _compute_receive_pattern_weights(self):
        """
        Extract 12 weights for each range bin using TA LUT, AC LUT, rd, and L0B starting range.
        Use rd, wd, wl, and L0B starting range to zero-out weights when beams are inactive.

        Returns:
        --------
        rx_weight_norm: 2D array of complex
            size = [num of chan x num of range bins], upsampled to DBF sampling frequency
        """

        rx_trm_info = self.rx_trm_info
        channels = rx_trm_info.channels
        num_chan = len(channels)
        slant_range = self.slant_range
        dbf_fs = rx_trm_info.dbf_fs
        adc_fs = rx_trm_info.adc_fs
        ta_lut_fs = rx_trm_info.ta_lut_fs
        num_lut_items = rx_trm_info.num_lut_items
        num_chan_qfsp = rx_trm_info.num_chan_qfsp
        wd = rx_trm_info.wd
        rd = rx_trm_info.rd
        wl = rx_trm_info.wl
        ac_chan_coef = rx_trm_info.ac_chan_coef
        ta_dbf_switch = rx_trm_info.ta_dbf_switch

        # Determine L0B sampling frequency
        # slant range is uniformly spaced
        data_fs = isce3.core.speed_of_light / (2 * slant_range.spacing)

        # Upsample to on-board DBF sampling rate
        num_range_bins = slant_range.size
        num_range_bins_dbf_fs = round(dbf_fs / data_fs * num_range_bins)
        rx_weight_norm = np.zeros((num_chan, num_range_bins_dbf_fs), dtype=np.complex)

        # Range sample index at onboard DBF processing sampling frequency with respect to transmit event.
        idx0_data = round(slant_range.first * 2 / isce3.core.speed_of_light * dbf_fs)

        # Compute rd, wd, and wl based on data sampling rate factor data_fs / adc_fs
        rd_dbf_fs = np.round(rd * dbf_fs / adc_fs).astype(int)
        wd_dbf_fs = np.round(wd * dbf_fs / adc_fs).astype(int)
        wl_dbf_fs = np.round(wl * dbf_fs / adc_fs).astype(int)

        # Compute weights of all 12 channels.
        for chan_idx, chan in enumerate(channels):
            # Determine which qFSP that the channel belongs to.
            # Channell indices read from L0B is one-based, needs to subtract one from it
            qfsp = int((chan - 1) / num_chan_qfsp)

            # Convert dbf_switch from 48 MHz to on-board DBF processing sampling rate
            dbf_switch = np.round(ta_dbf_switch[qfsp] * dbf_fs / ta_lut_fs)

            # Delay of this QFSP with respect to composite rangeline, in bins.
            qfsp_offset = round(rd_dbf_fs[chan_idx] - idx0_data)

            # Coefficient from AC table for this channel.
            acc = ac_chan_coef[chan_idx]

            # The first range bin at which 1st coeff in AC table will be applied
            start = max(0, qfsp_offset)

            for k in range(num_lut_items):
                #  last range bin at which k'th coeff in AC table will be applied
                stop = max(0, round(qfsp_offset + dbf_switch[k]))

                rx_weight_norm[chan_idx, start:stop] = acc[k]
                start = stop

            # Zero out coefficients when beams are inactive
            start_active = qfsp_offset + wd_dbf_fs[chan_idx]
            stop_active = start_active + wl_dbf_fs[chan_idx]

            rx_weight_norm[chan_idx, :start_active] = 0
            rx_weight_norm[chan_idx, stop_active:] = 0

        return rx_weight_norm, data_fs

    def _compute_transmit_pattern_weights(self):
        """
        Returns:
        -------
        tx_weight_norm: 2D array of complex
            normalized Tx beamforming weight, dim = [# of channel x # of HCAL pulses]
        """

        tx_trm = self.tx_trm_info
        channels = tx_trm.channels
        num_chan = len(channels)

        # Each range line contains HCAL, BCAL or LCAL measurements in the header
        # hcal_lines_idx, bcal_lines_idx, and lcal_lines_idx keep track of range line
        # numbers for each type of measurements in the L0B file.
        correlator_tap2 = tx_trm.correlator_tap2
        (
            hcal_lines_idx,
            lcal_lines_idx,
            bcal_lines_idx,
            num_lines,
        ) = self._compute_hcal_lines_idx_tx()

        cal_interval = bcal_lines_idx[1] - bcal_lines_idx[0]

        # Legacy L0B may not have BCAL_INTERVAL attribute, therefore it leads to
        # missing lcal_lines_idx and bcal_lines_idx. If so, Tx weights are of unity-gain.
        if (len(lcal_lines_idx) == 0) or (len(bcal_lines_idx) == 0):
            tx_weight_norm = np.ones(
                [num_chan, num_lines], dtype=np.complex_
            ) / np.sqrt(num_chan)
        else:
            # Use only Correlator 2nd tap
            num_bcal_lines_idx = len(bcal_lines_idx)
            num_hcal_lines_idx = len(hcal_lines_idx)

            correlator_tap2 = correlator_tap2.transpose()
            data_cal_div = np.zeros(correlator_tap2.shape, dtype=np.complex_)

            # TX complex divide: HCAL 2nd tap / BCAL 2nd tap (last BCAL before
            # current block of HCAL lines)
            # HCAL/BCAL is done on all HCAL lines. For BCAL range line, BCAL/BCAL
            # is performed, but discarded afterwards.

            for chan_idx, chan in enumerate(channels):
                start_idx = 0
                for i in range(num_bcal_lines_idx - 1):
                    end_idx = bcal_lines_idx[i + 1]
                    cal_div = (
                        correlator_tap2[chan_idx, start_idx:end_idx]
                        / correlator_tap2[chan_idx, bcal_lines_idx[i]]
                    )
                    data_cal_div[chan_idx, start_idx:end_idx] = cal_div

                    start_idx = end_idx

                # Last block of range lines that is outside of the loop above
                data_cal_div[chan_idx, bcal_lines_idx[-1] :] = (
                    correlator_tap2[chan_idx, bcal_lines_idx[-1] :]
                    / correlator_tap2[chan_idx, bcal_lines_idx[-1]]
                )

            # Only computed HCAL Line weights are kept
            tx_weight_norm = data_cal_div[:, hcal_lines_idx]

            # Power normalize tx weights to unity-gain
            for i in range(num_hcal_lines_idx):
                tx_norm_factor = np.sqrt(np.sum(np.abs(tx_weight_norm[:, i]) ** 2))
                tx_weight_norm[:, i] = tx_weight_norm[:, i] / tx_norm_factor

        return tx_weight_norm

    def _elaz2slantrange(self, pulse_time):
        """
        Interpolate antenna elevation angles to corresponding slant ranges:

        Parameters:
        -----------
        pulse_time: float or array of float
            transmit pulse time of range line in seconds w.r.t radar product reference

        Returns:
        --------
        slant_range_el: array of float
            Slant range computed at antenna pattern elevation angles
        """

        tx_trm = self.tx_trm_info
        el_angles_rad = self.el_ant_info.el_angles_rad
        az_cut_angles_rad = self.el_ant_info.az_cut_angles_rad
        fc = tx_trm.fc
        orbit = self.orbit
        dem = self.dem
        attitude = self.attitude
        sc_pos_vel = orbit.interpolate(pulse_time)
        sc_pos_ecef = sc_pos_vel[0]
        sc_vel_ecef = sc_pos_vel[1]

        q_ant2ecef = attitude.interpolate(pulse_time)

        wavelength = isce3.core.speed_of_light / fc

        # Convert Elevation angle to slant range
        slant_range_el, _, _ = ant.ant2rgdop(
            el_angles_rad,
            az_cut_angles_rad,
            sc_pos_ecef,
            sc_vel_ecef,
            q_ant2ecef,
            wavelength,
            dem,
        )

        return slant_range_el

    def _compute_hcal_lines_idx_tx(self):
        """
        Each rangeline contains HCAL, BCAL or LCAL measurement in the header.
        Based on cal_path_mask, determine rangeline numbers of each kind of 3 CAL measurements
        within the L0B file.

        Returns:
        -----------
        hcal_line_idx: array of int
            HCAL range line indices
        bcal_line_idx: array of int
            BCAL range line indices
        lcal_line_idx: array of int
            LCAL range line indices
        """

        tx_trm = self.tx_trm_info
        cal_path_mask = tx_trm.cal_path_mask
        num_lines = len(cal_path_mask)
        rng_lines_idx = np.arange(num_lines)

        hcal_lines_idx = rng_lines_idx[cal_path_mask == CalPath.HPA]

        lcal_lines_idx = rng_lines_idx[cal_path_mask == CalPath.LNA]
        bcal_lines_idx = rng_lines_idx[cal_path_mask == CalPath.BYPASS]

        return hcal_lines_idx, lcal_lines_idx, bcal_lines_idx, num_lines

    def apply_weights_to_beams_tx(self, pulse_time):
        """
        Sum and average the product of antenna gain and Tx weights,
        and interpolate beamformed gain pattern as a function of slant range
        pulse time is used to infer slant range from elevation angles.

        Parameters:
        -----------
        pulse_time: scalar or array of float
             transmit time of range line in seconds w.r.t radar product reference, single or multiple range lines

        Returns:
        --------
        bmf_sr_interp_tx: 1D or 2D array of complex
            interpolated Tx beamformed gain as a function of slant range
        """

        tx_trm = self.tx_trm_info
        channels = tx_trm.channels
        slant_range = self.slant_range
        fc = tx_trm.fc
        transmit_time = tx_trm.transmit_time
        tx_correction_factor = tx_trm.tx_correction_factor
        tx_weight_norm = self._tx_weight_norm

        ant_gain_el = self.el_ant_info.el_ant_gain
        el_angles_rad = self.el_ant_info.el_angles_rad
        az_cut_angles_rad = self.el_ant_info.az_cut_angles_rad
        num_chan = len(channels)
        slant_range_data_fs = np.asarray(slant_range)

        # Apply Temperture Phase Correction
        # tx_correction_factor is an array of complex numbers with size =  [num_chan]
        # The phase of the out of cal-loop components that are not part of HCAL/BCAL/LCAL measurements
        # may need to be accounted for. It could be through a calibration temperature correction table
        # lookup that contains phase offset due to temperature dependence of out-of-cal loop components
        # or through fixed phase offsets for each channel per polarization. This decision will be made
        # later. If it is decided that look up into calibration measurements tables is required,
        # it will be implemented later.  For R3.0, constant 1 is assumed for all channels
        # with a placeholder included in the code.

        bmf = np.zeros(tx_weight_norm.shape, dtype=complex)
        for chan_idx in range(num_chan):
            bmf[chan_idx] = tx_weight_norm[chan_idx] * tx_correction_factor[chan_idx]

        # Compute HCAL, LCAL, and BCAL line indices
        (
            hcal_lines_idx,
            lcal_lines_idx,
            bcal_lines_idx,
            num_lines,
        ) = self._compute_hcal_lines_idx_tx()

        assert tx_weight_norm.shape[0] == num_chan
        assert ant_gain_el.shape[0] == num_chan

        bmf = bmf.transpose() @ ant_gain_el

        # Check if pulse time input is a single float or a list of floats
        # pulse_time could be scalar or vector
        if np.isscalar(pulse_time):
            pulse_time = [pulse_time]

        bmf_sr_interp_tx = []
        for i, tx_time in enumerate(pulse_time):
            pulse_idx = np.where(transmit_time == pulse_time[i])[0]
            if pulse_idx in hcal_lines_idx:
                bmf_pulse = np.squeeze(bmf[pulse_idx])

                # Interpolate Tx BMF gain as a function of slant range
                slant_range_el = self._elaz2slantrange(pulse_time[i])
                bmf_sr_interp = np.interp(
                    slant_range_data_fs, slant_range_el, bmf_pulse
                )
                bmf_sr_interp_tx.append(bmf_sr_interp)
            else:
                raise ValueError("Pulse time input {} is not valid".format(pulse_time))

        bmf_sr_interp_tx = np.asarray(bmf_sr_interp_tx)

        return bmf_sr_interp_tx

    def apply_weights_to_beams_rx(self, pulse_time):
        """
        Sum and average the product of antenna gain and conjugate transpose
        of Rx weight, and interpolate beamformed gain pattern based on slant range
        Antenna gain will first be converted to slant ranges, sampled at beamforming frequency
        of 240 MHz. The final Rx beamformed pattern is interpolated to data sampling frequency.

        Parameters:
        -----------
        pulse_time: scalar or array of float
            transmit time of range line in seconds w.r.t radar product reference, single or multiple range lines

        Returns:
        --------
        bmf_sr_interp_rx: 1D or 2D array of complex
            interpolated Tx beamformed gain as a function of slant range
        slant_range_el: array of float
            interpolated slant range of elevation angles
        """

        rx_weight_norm = self._rx_weight_norm
        rx_trm = self.rx_trm_info
        rx_correction_factor = rx_trm.rx_correction_factor
        num_chan = len(rx_trm.channels)
        ant_gain_el = self.el_ant_info.el_ant_gain
        slant_range = self.slant_range
        slant_range_data_fs = np.asarray(slant_range)
        dbf_fs = rx_trm.dbf_fs
        data_fs = self._data_fs

        # Determine number of range bins after upsampling to DBF sampling frequency
        num_range_bins_dbf_fs = len(rx_weight_norm[0])

        # Check if pulse time input is a single float or a list of floats
        if isinstance(pulse_time, float):
            pulse_time = [pulse_time]

        # Multiply by possible phase offset fudge factor
        rx_weight_corrected = np.zeros(rx_weight_norm.shape, dtype=complex)
        for chan_idx in range(num_chan):
            rx_weight_corrected[chan_idx] = (
                rx_weight_norm[chan_idx] * rx_correction_factor[chan_idx]
            )

        # Compute product of beam weights and antenna gain
        # Interpolate Antenna gain from angle to slant range in DBF processing sampling frequency
        # Compute slant range spacing based on upsampling factor data_fs / dbf_fs
        slant_range_spacing_dbf_fs = slant_range.spacing * data_fs / dbf_fs
        slant_range_interp_dbf_fs = (
            slant_range.first
            + np.arange(num_range_bins_dbf_fs) * slant_range_spacing_dbf_fs
        )

        bmf_sr_interp_rx = []
        for i, tx_time in enumerate(pulse_time):
            rx_bf_gain_composite = 0
            slant_range_el = self._elaz2slantrange(pulse_time[i])
            for chan in range(num_chan):
                # Interpolate elevation antenna gain of each channel to slant range at DBF sampling rate
                ant_gain_sr_interp_ta_fs = np.interp(
                    slant_range_interp_dbf_fs, slant_range_el, ant_gain_el[chan]
                )

                # Multiply weights by antenna gain of each channel in time (slant range) domain
                rx_bf_gain = rx_weight_corrected[chan] * ant_gain_sr_interp_ta_fs

                # Downsample and interpolate Rx dbf gain to slant range of data sampling rate
                if dbf_fs % data_fs == 0:
                    rx_bf_gain = rx_bf_gain[:: int(dbf_fs / data_fs)]
                else:
                    rx_bf_gain = np.interp(
                        slant_range, slant_range_interp_ta_fs, rx_bf_gain
                    )
                rx_bf_gain_composite += rx_bf_gain

            bmf_sr_interp_rx.append(rx_bf_gain_composite)

        bmf_sr_interp_rx = np.asarray(bmf_sr_interp_rx)

        return bmf_sr_interp_rx, slant_range_el

    def form_two_way(self, pulse_time):
        """
        Multiply Tx and Rx beamformed antenna patterns to
        compute the resulting combined beamformed gain pattern

        Parameters:
        -----------
        pulse_time: scalar or array of float
            transmit time of range line in seconds w.r.t radar product reference, single or multiple range lines

        Returns:
        --------
        bmf_sr_interp: 1D or 2D array of complex
            interpolated beamformed gain as a function of slant range at data sampling rate
        bmf_sr_interp_tx: 1D or 2D array of complex
            interpolated Tx beamformed gain as a function of slant range at data sampling rate
        bmf_sr_interp_rx: 1D or 2D array of complex
            interpolated Rx beamformed gain as a function of slant range at data sampling rate
        slant_range_el: array of float
            Slant range computed at antenna pattern elevation angles
        """

        bmf_sr_interp_tx = self.apply_weights_to_beams_tx(pulse_time)
        bmf_sr_interp_rx, slant_range_el = self.apply_weights_to_beams_rx(pulse_time)

        # Compute product of combined Tx and Rx gain pattern
        bmf_sr_interp = bmf_sr_interp_tx * bmf_sr_interp_rx

        return bmf_sr_interp, bmf_sr_interp_tx, bmf_sr_interp_rx, slant_range_el
