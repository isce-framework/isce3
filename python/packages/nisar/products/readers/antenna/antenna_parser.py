from __future__ import annotations

import h5py
import numpy as np
import re
from dataclasses import dataclass
import typing

from isce3 import antenna as ant


@dataclass
class AntPatCut:
    """
    Antenna pattern cut(s) information, either relative or absolute patterns,
    in either elevation (EL) or azimuth (AZ) direction for single beam or
    multiple-beam antenna system.

    Attributes
    ----------
    cut_angle : float
        AZ/EL angle in radians for EL/AZ cut(s).
    angle : np.ndarray(float)
        EL/AZ angles in radians common among all beams if multi-beam.
    copol_pattern : np.ndarray(float or complex)
        1-D or 2-D array of Co-pol EL/AZ complex or real amplutide pattern(s).
        Unit is (V/m) for absolute pattern (electric fields) and
        unitless otherwise.
        In case of multi-beam, the array is 2-D with shape (beams, angles).
    cxpol_pattern : np.ndarray(float or complex) or None, optional
        1-D or 2-D array of X-pol EL/AZ complex or real amplutide pattern(s).
        Unit is (V/m) for absolute pattern (electric fields) and
        unitless otherwise.
        In case of multi-beam, the array is 2-D with shape (beams, angles).

    """
    cut_angle: float
    angle: np.ndarray
    copol_pattern: np.ndarray
    cxpol_pattern: typing.Optional[np.ndarray] = None


class AntennaParser:
    """Class for parsing NISAR Antenna HDF5 file.

    Parameters
    ----------
    filename : str
        filename of HDF5 antenna file.

    Attributes
    ----------
    filename : str
        Filename of HDF5 antenna file.
    fid : h5py.File
        File object of HDF5.
    frequency : float
        RF frequency in Hz.
    frame : isce3.antenna.frame
        isce3 Frame object.
    rx_beams : list of str
        List of names of all receive beams.
    tag : str
        Tag name for antenna patterns.
    timestamp : str
        Time stamp in UTC.
    tx_beams : list of str
        List of names of all transmit beams.
    version : str
        Version of the file.

    Raises
    ------
    IOError
        If HDF5 filename does not exist or can not be opened.

    """

    def __init__(self, filename):
        self._filename = filename
        self._fid = h5py.File(filename, mode='r', libver='latest',
                              swmr=True)

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
    def rx_beams(self):
        return [item for item in self._fid if "RX" in item.upper()]

    @property
    def tx_beams(self):
        return [item for item in self._fid if "TX" in item.upper()]

    @property
    def frequency(self):
        return self._from_cut_attrs("frequency")

    @property
    def timestamp(self):
        tm = self._from_cut_attrs("timestamp")
        try:
            return tm.decode()
        except AttributeError:
            return tm

    @property
    def version(self):
        vr = self._from_cut_attrs("version")
        try:
            return vr.decode()
        except AttributeError:
            return vr

    @property
    def tag(self):
        tg = self._from_cut_attrs("tag")
        try:
            return tg.decode()
        except AttributeError:
            return tg

    @property
    def frame(self):
        return ant.Frame(self._gridtype())

    def beam_numbers(self, pol='H'):
        """List of all RX beam numbers for a desired pol

        Parameters
        ----------
        pol : str, default='H'
            Polarization of the beam , either
            `H` or `V'. It is case insensitive.

        Returns
        -------
        list of int

        """
        pol = self._check_pol(pol)
        re_pat = self._form_rx_regpat(pol)
        return [int(re_pat.fullmatch(grp).group()[2:4]) for grp in
                self._fid if re_pat.fullmatch(grp)]

    def num_beams(self, pol='H'):
        """Number of individual [RX] beams for each pol.

        Parameters
        ----------
        pol : str, default='H'
            Polarization of the beam , either
            `H` or `V'. It is case insensitive.

        Returns
        -------
        int
            Number of beams for the `pol`.

        Raises
        ------
        ValueError
            For bad `pol` value.

        """
        pol = self._check_pol(pol)
        return len([rx for rx in self.rx_beams if 'DBF' not in
                    rx.upper() and pol in rx.upper()])

    def locate_beams_peak(self, pol='H'):
        """Get peak locations in EL direction for beams.

        Use a 2nd-order polyfit around estimated peak to get the exact peak
        locations in EL direction.

        Returns
        -------
        np.ndarray(float)
            EL angles of the peak in radians.
        float
            Common azimuth angle of EL-cut peaks in radians.

        """
        num_beams = self.num_beams(pol)
        el_loc = np.zeros(num_beams, dtype='f8')
        az_ang = 0.0
        for beam in range(num_beams):
            ant_el = self.el_cut(beam + 1, pol)
            # get approximate peak power locaton first
            pow_db = 20 * np.log10(abs(ant_el.copol_pattern))
            idx_pk = np.nanargmax(pow_db)
            # pick 5 points rather than 3, two extra just in case.
            # perform second-order polyfit (gain in dB versus EL angle in rad)
            # and then find the peak where the first derivative is zero.
            el_slice = slice(max(idx_pk - 2, 0),
                             min(idx_pk + 3, ant_el.angle.size))
            pf_coef = np.polyfit(ant_el.angle[el_slice], pow_db[el_slice],
                                 deg=2)
            el_loc[beam] = - pf_coef[1] / (2 * pf_coef[0])
            # get averaged azimuth angles among all EL-cut patterns
            az_ang += ant_el.cut_angle
        az_ang /= num_beams
        return el_loc, az_ang

    def locate_beams_overlap(self, pol='H'):
        """Get overlap location of adjacent beams in EL direction.

        The number of locations is equal to number of beams minus one.
        In case of single beam, None will return.

        These locations in EL direction are useful for single-tap DBF,
        relative antenna pattern estimation, etc.

        Parameters
        ----------
        pol : str, default='H'
            Polarization of the beam , either `H` or `V'.
            It is case insensitive.

        Returns
        -------
        np.ndarray(float) or None
            `N-1` EL angles in radians for N beam. In case of N=1,
            None will be returned.
        float, optional
            Common azimuth angle of EL-cut peaks in radians. Only
            returned if N>1.

        Raises
        ------
        RuntimeError
            If number of transition points is not equal the total beams
            minus one.
            If the transition points are not monotonically increasing

        Notes
        -----
        A linear interpolation of envelope of EL pattern is performed if
        EL spacing is greater than 20 mdg. The expected accuracy at the
        beam transition in EL shall be within around +/- 10 mdeg.

        """
        # required EL spacing to be at least 20 mdeg
        el_spacing_min = np.deg2rad(20e-3)
        ant_el = self.el_cut_all(pol)
        # check wether it is single beam or multiple beam
        num_beams = ant_el.copol_pattern.shape[0]
        if num_beams == 1:
            return None
        # multi beam
        # find dominant beams per max absolute amplitude or power
        amp_pats_el = abs(ant_el.copol_pattern)
        idx_pk_first = amp_pats_el[0].argmax()
        idx_pk_last = amp_pats_el[-1].argmax()
        # EL slice within peak of the first beam and peak of the last beam
        slice_el = slice(idx_pk_first, idx_pk_last + 1)
        amp_pats_el = amp_pats_el[:, slice_el]
        # perform linear interpolation if el spacing is larger than
        # required "el_spacing_min"
        d_el = np.diff(ant_el.angle).mean()
        if d_el > el_spacing_min:
            el = ant_el.angle[slice_el]
            num_el_int = round((el[-1] - el[0]) / el_spacing_min) + 1
            el_int = np.linspace(el[0], el[-1], num=num_el_int)
            amp_pats_int = np.zeros((num_beams, num_el_int), dtype='f8')
            for nn in range(num_beams):
                amp_pats_int[nn] = np.interp(el_int, el, amp_pats_el[nn])
        else:
            amp_pats_int = amp_pats_el
            el_int = ant_el.angle[slice_el]
        idx_max = amp_pats_int.argmax(axis=0)
        idx_trans = np.where(np.diff(idx_max) == 1)[0]
        if idx_trans.size != num_beams - 1:
            raise RuntimeError(f'Expected {num_beams - 1} transition '
                               f'points but got {idx_trans.size} with '
                               f'EL index values {idx_trans}!')
        if not np.all(sorted(set(idx_trans)) == idx_trans):
            raise RuntimeError(
                f'Transition points {idx_trans} are not monotonically '
                'increasing!')
        # EL angle at transition point where two adjacent beams are
        # equally dominant
        ela_trans = 0.5 * (el_int[idx_trans] + el_int[idx_trans + 1])
        return ela_trans, ant_el.cut_angle

    def el_cut(self, beam=1, pol='H', full=False):
        """Parse an Elevation cut pattern from a `RX` beam.

        Parse individual RX Elevation-cut 1-D pattern for
        a desired polarizatrion `pol` and beam number `beam`.

        Parameters
        ----------
        beam : int, default=1
            Beam number starting from one.
        pol : str, default='H'
            Polarization of the beam , either
            `H` or `V'. It is case insensitive.
        full : bool, default=False
            Whether or not to return the full angular coverage of input data.
            If False, it will limit angular coverage to within around 10 dB
            dynamic range w.r.t the one-way peak magnitude of the beam.

        Returns
        -------
        AntPatCut
            angle : np.ndarray (float or complex)
                Elevation angles in radians.
            copol_pattern : np.ndarray (float or complex)
                Co-pol 1-D elevation pattern in V/m.
            cxpol_pattern : np.ndarray (float or complex)
                Cross-pol 1-D elevation pattern in V/m.
                None if there no cx-pol pattern!
            cut_angle : float
                Azimuth angle in radians for obtaining elevation cut.

        Raises
        ------
        ValueError
            For bad input arguments
        RuntimeError
            For missing fields/attributes in HDF5

        """
        return self._get_ang_cut(beam, pol, 'elevation', full=full)

    def az_cut(self, beam=1, pol='H', full=False):
        """Parse an Azimuth cut pattern from a `RX` beam.

        Parse individual RX Azimuth-cut 1-D pattern for
        a desired polarizatrion `pol` and beam number `beam`.

        Parameters
        ----------
        beam : int, default=1
            Beam number starting from one.
        pol : str, default='H'
            Polarization of the beam , either
            `H` or `V'. It is case insensitive.
        full : bool, default=False
            Whether or not to return the full angular coverage of input data.
            If False, it will limit angular coverage to within around 10 dB
            dynamic range w.r.t the one-way peak magnitude of the beam.

        Returns
        -------
        AntPatCut
            angle : np.ndarray (float or complex)
                Azimuth angles in radians.
            copol_pattern : np.ndarray (float or complex)
                Co-pol 1-D azimuth pattern in V/m.
            cxpol_pattern : np.ndarray (float or complex)
                Cross-pol 1-D azimuth pattern in V/m.
                None if there no cx-pol pattern!
            cut_angle : float
                Elevation angle in radians for obtaining azimuth cut.

        Raises
        ------
        ValueError
            For bad input arguments
        RuntimeError
            For missing fields/attributes in HDF5

        """
        return self._get_ang_cut(beam, pol, 'azimuth', full=full)

    def el_cut_all(self, pol='H', full=False):
        """Parse all Co-pol and Cx-pol EL (Elevation) cuts.

        Get all uniformly-spaced EL cuts of co-pol and cx-pol and store them in
        a matrix with shape `num_beams` by `number of angles`. The number of
        uniformly-spaced angles is determined by min, max angles from first
        and last beams and the spacing from the first beam.

        Parameters
        ----------
        pol : str, default='H'
            Polarization , either 'H' or 'V'. It is case-insensitive!
        full : bool, default=False
            Whether or not to return the full angular coverage of input data.
            If False, it will limit angular coverage to within around 10 dB
            dynamic range on both ends, first and last beam, w.r.t the
            one-way peak magnitude of the corresponding beam.

        Returns
        -------
        AntPatCut
            angle : np.ndarray (float)
                Uniformly-spaced elevation angles in radians.
            copol_pattern : np.ndarray (float or complex)
                Interpolated co-pol 1-D elevation pattern in V/m with
                shape (number-of-beams, number-of-EL-angles).
            cxpol_pattern : np.ndarray (float or complex)
                Interpolated cx-pol 1-D elevation pattern in V/m with
                shape (number-of-beams, number-of-EL-angles).
            cut_angle : float
                Mean azimuth angle in radians from which
                elevation patterns are obtained.

        Raises
        ------
        ValueError
            For bad `pol` value.
        KeyError
            For missing mandatory fields in the product.

        Notes
        -----
        Cx-pol patterns will be set to zeros with the same shape as co-pol
        ones if the cx-pol patterns are missing in the product.

        """
        return self._ang_cut_all(cut_type='elevation', pol=pol, full=full)

    def az_cut_all(self, pol='H', full=False):
        """Parse all Co-pol and Cx-pol AZ (Azimuth) cuts.

        Get all uniformly-spaced AZ cuts of co-pol and cx-pol and store them in
        a matrix with shape `num_beams` by `number of angles`. The number of
        uniformly-spaced angles is determined by min, max angles from first
        and last beams and the spacing from the first beam.

        Parameters
        ----------
        pol : str, default='H'
            Polarization , either 'H' or 'V'. It is case-insensitive!
        full : bool, default=False
            Whether or not to return the full angular coverage of input data.
            If False, it will limit angular coverage to within around 10 dB
            dynamic range on both ends, first and last beam, w.r.t the
            one-way peak magnitude of the corresponding beam.

        Returns
        -------
        AntPatCut
            angle : np.ndarray (float)
                Uniformly-spaced azimuth angles in radians.
            copol_pattern : np.ndarray (float or complex)
                Interpolated co-pol 1-D azimuth pattern in V/m with
                shape (number-of-beams, number-of-AZ-angles).
            cxpol_pattern : np.ndarray (float or complex)
                Interpolated cx-pol 1-D azimuth pattern in V/m with
                shape (number-of-beams, number-of-AZ-angles).
            cut_angle : float
                Elevation angle in radians from which
                azimuth patterns are obtained. Note that this value
                simply represents a mean among all EL angles at which
                individual AZ cuts are obtained.

        Raises
        ------
        ValueError
            For bad `pol` value.
        KeyError
            For missing mandatory fields in the product.

        Notes
        -----
        Cx-pol patterns will be set to zeros with the same shape as co-pol
        ones if the cx-pol patterns are missing in the product.

        """
        return self._ang_cut_all(cut_type='azimuth', pol=pol, full=full)

    def cut_angles_az_cuts(self, pol='H'):
        """
        Get array of elevation angles at which all individual AZ cuts are
        obtained per desired pol.

        Parameters
        ----------
        pol : str, default='H'
            Polarization , either 'H' or 'V'. It is case-insensitive!

        Returns
        -------
        np.ndarray(float)
            Elevation angles in radians for all azimuth cuts.

        """
        num_beam = self.num_beams(pol)
        cut_ang_all = np.zeros(num_beam)
        for nn in range(num_beam):
            grp_cut = self._fid[f'RX{nn+1:02d}{pol}/azimuth']
            cut_ang_all[nn] = grp_cut["angle"].attrs.get('cut_angle')
        return cut_ang_all

    # Helper functions listed below this line
    def _form_rx_regpat(self, pol: str) -> re.match:
        """Form a regular expression pattern for RX."""
        return re.compile(f'RX[0-9][0-9][{pol.upper()}]')

    def _check_pol(self, pol: str) -> str:
        """Check and get upper-case polarization type.

        """
        pol = pol.upper()
        if pol not in ['H', 'V']:
            raise ValueError("'pol' shall be either 'H' or 'V'")
        return pol

    def _from_cut_attrs(self, attr_name: str):
        """Get a value from cut attribute.

        """
        first_cut_name = list(self._fid[self.rx_beams[0]])[0]
        cut_attr_obj = self._fid[self.rx_beams[0] + '/'
                                 + first_cut_name].attrs
        try:
            return cut_attr_obj[attr_name]
        except KeyError:
            raise RuntimeError(
                f"'{attr_name}' not found in attribute of '{first_cut_name}'!")

    def _get_ang_cut(self, beam: int, pol: str, cut_name: str, *,
                     full: bool = False,
                     out_keys: tuple = ("angle", "copol_pattern",
                                        "cxpol_pattern"),
                     ang_attr: str = "cut_angle") -> AntPatCut:
        """Get angle and co/cross 1-D patterns.

        Parameters
        ----------
        beam : int
            Beam number starting from 1.
        pol : str
            Polarization, either "H" or "V"
        cut_name : str
            Name of the principal cut. Either "azimuth" or "elevation"
        full: bool, default=False
            If False, the angles on either sides will be truncated
            within one-way 10-dB beamwidth, otherwise the entire
            angular coverage will be returned.
        out_keys : tuple of str,
            default=("angle", "copol_pattern", "cxpol_pattern").
            Keys related to fieldnames in HDF5 antenna file that
            shall be stored in antenna cut dataclass "AntPatCut".
        ang_attr : str, default="cut_angle"
            Name of a desired attribute for cut angle in `angle` dataset
            of HDF5 antenna file.

        Returns
        -------
        nisar.products.readers.antenna.AntPatCut

        """
        pol = self._check_pol(pol)
        # get number of beams
        num_beam = self.num_beams(pol)
        if num_beam == 0:
            raise ValueError(f"There is no individual pattern for pol {pol}")
        if (beam < 1 or beam > num_beam):
            raise ValueError(f"'beam' shall be within [1,{num_beam}]")
        grp_cut = self._fid[f'RX{beam:02d}{pol}/{cut_name}']
        out = dict.fromkeys(out_keys)
        for key in out_keys:
            # other fields except "cxpol_pattern" are mandatory!
            try:
                fld = grp_cut[key]
            except KeyError:
                if key != "cxpol_pattern":
                    raise
                continue
            else:
                out[key] = fld[()]
            if ang_attr and key == "angle":
                out[ang_attr] = grp_cut[key].attrs.get(ang_attr)

        # check for full or truncated angular coverage
        cut = AntPatCut(**out)
        if not full:
            _, _, slice_ang = xdb_points_from_cut(cut)
            cut.angle = cut.angle[slice_ang]
            cut.copol_pattern = cut.copol_pattern[slice_ang]
            if cut.cxpol_pattern is not None:
                cut.cxpol_pattern = cut.cxpol_pattern[slice_ang]
        return cut

    def _gridtype(self) -> str:
        """Get spherical grid type.

        """
        grd = self._from_cut_attrs("grid_type")
        try:
            return grd.decode().replace('-', '_')
        except AttributeError:
            return grd.replace('-', '_')

    def _ang_cut_all(self, cut_type: str, pol: str, full: bool = False
                     ) -> AntPatCut:
        """
        Get all co-pol and cx-pol cut patterns of a certain cut_type
        and pol.

        Parameters
        ----------
        cut_type : str
            either "elevation" or "azimuth"
        pol : str
            Polarization
        full: bool, default=False
            If False, the angles on either sides will be truncated
            within one-way 10-dB beamwidth, otherwise the entire
            angular coverage will be returned.

        Returns
        -------
        nisar.products.readers.antenna.AntPatCut

        """
        num_beam = self.num_beams(pol)
        # determine full angular coverage with uniform spcaing over all beams
        beam_first = self._get_ang_cut(
            1, pol, cut_type, full=True, out_keys=("angle", "copol_pattern"))
        if num_beam > 1:
            beam_last = self._get_ang_cut(
                num_beam, pol, cut_type, full=True, out_keys=("angle",
                                                              "copol_pattern"))
        else:
            beam_last = beam_first
        # check for angle coverage, full or truncated
        if full:
            ang_first = beam_first.angle[0]
            ang_last = beam_last.angle[-1]
        else:  # limit angluar coverage
            ang_first, _, _ = xdb_points_from_cut(beam_first)
            _, ang_last, _ = xdb_points_from_cut(beam_last)
        ang_spacing = beam_first.angle[1] - beam_first.angle[0]
        num_ang = int(np.ceil((ang_last - ang_first) / ang_spacing)) + 1
        # linearly interpolate each beam over desired angular coverage with
        # out-of-range values filled with float or complex zero.
        out = {}
        out["angle"] = np.linspace(ang_first, ang_last, num_ang)
        out["copol_pattern"] = np.zeros((num_beam, num_ang),
                                        beam_first.copol_pattern.dtype)
        out["cxpol_pattern"] = np.zeros_like(out["copol_pattern"])
        out_of_range_val = 0.0
        cut_ang_ave = 0.0

        for nn in range(num_beam):
            beam = self._get_ang_cut(nn + 1, pol, cut_type, full=True)
            out["copol_pattern"][nn, :] = np.interp(
                out["angle"], beam.angle, beam.copol_pattern,
                left=out_of_range_val, right=out_of_range_val)
            if beam.cxpol_pattern is not None:
                out["cxpol_pattern"][nn, :] = np.interp(
                    out["angle"], beam.angle, beam.cxpol_pattern,
                    left=out_of_range_val, right=out_of_range_val)
            cut_ang_ave += beam.cut_angle

        out["cut_angle"] = cut_ang_ave / num_beam
        return AntPatCut(**out)


def xdb_points_from_cut(cut: AntPatCut, x_db: float = 10.0
                        ) -> tuple[float, float, slice]:
    """
    Get approximate angles (radians) within x-dB dynamic range of the
    peak of a cut pattern in EL or AZ.

    Parameters
    ----------
    cut : nisar.products.readers.antenna.AntPatCut
        Single-beam cut pattern info.
    x_db : float, default=10.0
        x dB below the peak.
        Assumed this level is above the largest side lobe.

    Returns
    -------
    float
        Approximate x-dB below the peak on the left side of the peak
        in radians
    float
        Approximate x-dB below the peak on the right side of the peak
        in radians
    slice
        angle index slice for x-dB beamwidth.

    Notes
    -----
    The code tries to find the first left/right angles on either side
    of the peak to be at least `x_db` below the peak value provided enough
    angular margins on either sides.

    """
    x = 10 ** (-abs(x_db) / 20.)
    # ignore cx-pol pattern
    amp_pat = abs(cut.copol_pattern)
    idx_pk = np.argmax(amp_pat)
    pk = amp_pat[idx_pk]
    thrs = x * pk
    # left side of the peak
    idx_left = abs(amp_pat[:idx_pk] - thrs).argmin()
    # If possible adjust the left index to make sure its mag is
    # at least x-dB below the peak
    if idx_left > 0 and amp_pat[idx_left] > thrs:
        # check gain in case it is not monotonically decreasing
        if not (amp_pat[idx_left - 1] > thrs):
            idx_left -= 1
    # right side of the peak
    idx_right = abs(amp_pat[idx_pk:] - thrs).argmin() + idx_pk
    # If possible adjust the right index to make sure its mag is
    # at least x-dB below the peak
    if idx_right < (amp_pat.size - 1) and amp_pat[idx_right] > thrs:
        # check gain in case it is not monotonically decreasing
        if not (amp_pat[idx_right + 1] > thrs):
            idx_right += 1

    return (cut.angle[idx_left], cut.angle[idx_right],
            slice(idx_left, idx_right + 1))
