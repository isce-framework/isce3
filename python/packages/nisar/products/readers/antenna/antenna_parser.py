#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import collections as cl
import numpy as np

from pybind_isce3 import antenna as ant


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
        Isce3 Frame object.
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

    def el_cut(self, beam=1, pol='H'):
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

        Returns
        -------
        cl.namedtuple
            angle : np.ndarray (float or complex) 
                Elevation angles in radians.
            copol_pattern : np.ndarray (float or complex) 
                Co-pol 1-D elevation pattern in V/m.
            cxpol_pattern : np.ndarray (float or complex) 
                Cross-pol 1-D elevation pattern in V/m. 
                None if there no x-pol pattern!
            cut_angle : float 
                Azimuth angle in radians for obtaining elevation cut.

        Raises
        ------
        ValueError 
            For bad input arguments
        RuntimeError 
            For missing fields/attributes in HDF5

        """
        return self._get_ang_cut(beam, pol, 'elevation')

    def az_cut(self, beam=1, pol='H'):
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

        Returns
        -------
        cl.namedtuple
            angle : np.ndarray (float or complex) 
                Azimuth angles in radians.
            copol_pattern : np.ndarray (float or complex) 
                Co-pol 1-D azimuth pattern in V/m.
            cxpol_pattern : np.ndarray (float or complex) 
                Cross-pol 1-D azimuth pattern in V/m. 
                None if there no x-pol pattern!
            cut_angle : float 
                Elevation angle in radians for obtaining azimuth cut.

        Raises
        ------
        ValueError 
            For bad input arguments
        RuntimeError 
            For missing fields/attributes in HDF5        

        """
        return self._get_ang_cut(beam, pol, 'azimuth')

    def el_cut_all(self, pol='H'):
        """Parse all Co-pol EL cuts.

        Get all uniformly-spaced EL cuts of co-pol store them in a matrix
        with shape `num_beams` by `number of angles`. The number of
        uniformly-spaced angles is determined by min, max angles from first 
        and last beams and the spacing from the first beam.             

        Parameters
        ----------
        pol : str, default='H' 
            Polarization , either 'H' or 'V'. It is case-insensitive!

        Returns
        -------
        cl.namedtuple
            angle : np.ndarray (float)
                Uniformly-spaced elevation angles in radians.
            copol_pattern : np.ndarray (float or complex) 
                Interpolated co-pol 1-D elevation pattern in V/m with 
                shape (number-of-beams, number-of-EL-angles). 
            cut_angle : float 
                Mean azimuth angle in radians from which 
                elevation patterns are obtained.

        Raises
        ------
        ValueError
            For bad `pol` value.

        """
        num_beam = self.num_beams(pol)
        # determine full angular coverage with uniform spcaing over all beams
        beam_first = self._get_ang_cut(
            1, pol, 'elevation', out_keys=("angle", "copol_pattern"))
        if num_beam > 1:
            beam_last = self._get_ang_cut(
                num_beam, pol, 'elevation', out_keys=("angle",))
        else:
            beam_last = beam_first
        num_ang = int(np.ceil((beam_last.angle[-1] - beam_first.angle[0]) / (
            beam_first.angle[1] - beam_first.angle[0]))) + 1
        # linearly interpolate each beam over full angular coverage with out
        # of range values filled with float or complex zero.
        out = {}
        out["angle"] = np.linspace(
            beam_first.angle[0], beam_last.angle[-1], num_ang)
        out["copol_pattern"] = np.zeros((num_beam, num_ang),
                                        beam_first.copol_pattern.dtype)
        out_of_range_val = 0.0
        cut_ang_ave = 0.0

        for nn in range(num_beam):
            beam = self._get_ang_cut(
                nn + 1, pol, 'elevation', out_keys=("angle", "copol_pattern"))
            out["copol_pattern"][nn, :] = np.interp(
                out["angle"], beam.angle, beam.copol_pattern,
                left=out_of_range_val, right=out_of_range_val)
            cut_ang_ave += beam.cut_angle

        out["cut_angle"] = cut_ang_ave / num_beam
        return cl.namedtuple('el_cut', out)(*out.values())

    # Helper functions listed below this line

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

    def _get_ang_cut(self, beam: int, pol: str, cut_name: str,
                     out_keys: tuple = ("angle", "copol_pattern",
                                        "cxpol_pattern"),
                     ang_attr: str = "cut_angle") -> cl.namedtuple:
        """Get angle and co/cross 1-D patterns.

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
            out[key] = grp_cut.get(key)[()]
            if ang_attr and key == "angle":
                out[ang_attr] = grp_cut[key].attrs.get(ang_attr)
        return cl.namedtuple(cut_name+'_cut', out)(*out.values())

    def _gridtype(self) -> str:
        """Get spherical grid type.

        """
        grd = self._from_cut_attrs("grid_type")
        try:
            return grd.decode().replace('-', '_')
        except AttributeError:
            return grd.replace('-', '_')
