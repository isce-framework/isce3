from datetime import datetime, timezone
import h5py
import logging
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose
import os
from typing import Optional
import isce3
from isce3.core import LUT2d, DateTime, Orbit, Attitude, Quaternion, Ellipsoid
from isce3.product import RadarGridParameters
from isce3.geometry import DEMInterpolator
from isce3.product import get_radar_grid_nominal_ground_spacing
from nisar.h5 import set_string
from isce3.core.types import complex32
from nisar.products import descriptions
from nisar.products.readers.Raw import Raw
from nisar.products.readers.rslc_cal import RslcCalibration
from nisar.workflows.h5_prep import add_geolocation_grid_cubes_to_hdf5
from nisar.workflows.compute_stats import write_stats_complex_data


log = logging.getLogger("SLCWriter")

# TODO refactor isce::io::setRefEpoch
def time_units(epoch: DateTime) -> str:
    # XXX isce::io::*RefEpoch don't parse or serialize fraction.
    if epoch.frac != 0.0:
        raise ValueError("Reference epoch must have integer seconds.")
    date = "{:04d}-{:02d}-{:02d}".format(epoch.year, epoch.month, epoch.day)
    time = "{:02d}:{:02d}:{:02d}".format(epoch.hour, epoch.minute, epoch.second)
    return "seconds since " + date + "T" + time


def assert_same_lut2d_grid(x: np.ndarray, y: np.ndarray, lut: LUT2d):
    assert_allclose(x[0], lut.x_start)
    assert_allclose(y[0], lut.y_start)
    assert (len(x) > 1) and (len(y) > 1)
    assert_allclose(x[1] - x[0], lut.x_spacing)
    assert_allclose(y[1] - y[0], lut.y_spacing)
    assert lut.width == len(x)
    assert lut.length == len(y)


def add_cal_layer(group: h5py.Group, lut: LUT2d, name: str,
                  epoch: DateTime, units: str, description: str) -> None:
    """Add calibration LUT to HDF5 group, making sure that its domain matches
    any existing cal LUTs.

    Parameters
    ----------
    group : h5py.Group
        Group where LUT and its axes will be stored.
    lut : isce3.core.LUT2d
        Look up table.  Axes are assumed to be y=zeroDopplerTime, x=slantRange.
    name : str
        Name of dataset to store LUT data (can be path relative to `group`).
    epoch: isce3.core.DateTime
        Reference time associated with y-axis.
    units : str
        Units of lut.data.  Will be stored in `units` attribute of dataset.
    description : str
        Description to be stored in the `description` attribute of the dataset.
    """
    # If we've already saved one cal layer then we've already saved its
    # x and y axes.  Make sure they're the same and just save the new z data.
    xname, yname = "slantRange", "zeroDopplerTime"
    if (xname in group) and (yname in group):
        x, y = group[xname], group[yname]
        assert_same_lut2d_grid(x, y, lut)
        extant_epoch = isce3.io.get_ref_epoch(y.parent, y.name)
        assert extant_epoch == epoch
        data = lut.data
        z = group.require_dataset(name, data.shape, data.dtype)
        z[...] = data
        if description is not None:
            z.attrs["description"] = np.bytes_(description)
    elif (xname not in group) and (yname not in group):
        lut.save_to_h5(group, name, epoch, units)
        # C++ API doesn't add descriptions...
        group[xname].attrs["description"] = np.bytes_("Slant range "
            "dimension corresponding to processing information records")
        group[yname].attrs["description"] = np.bytes_("Zero doppler time "
            "dimension corresponding to processing information records")
        group[name].attrs["description"] = np.bytes_(description)
    else:
        raise IOError(f"Found only one of {xname} or {yname}."
                      "  Need both or none.")


def write_dataset(group: h5py.Group, name: str, dtype: np.dtype, value,
                  description: str, units: Optional[str] = None,
                  **kwargs) -> h5py.Dataset:
    """Write a dataset to a NISAR HDF5 product.

    Parameters
    ----------
    group : h5py.Group
        Handle to the HDF5 group object to write to.
    name : str
        Name of the dataset.
    dtype : numpy.dtype
        Element data type to write.  Must be numpy.bytes_ (or numpy.string_) for
        scalar string value or list of string values.
    value : object
        Object to write.  May be scalar or array. Will be converted to dtype if
        necessary.  Since HDF5 does not support zero-length strings, an empty
        string will be written as a single null character (which h5py reads back
        as an empty string).
    description : str
        Description of dataset to be stored in its `description` attribute.
        Will be converted to a byte array.
    units : str, optional
        Physical units corresponding to dataset to be stored in its `units`
        attribute.  Should be compatible with udunits2.  Dimensionless physical
        quantities should use units="1".  If None, no units attribute is added.
        Will be converted to a byte array.
    **kwargs
        Dataset creation options forwarded to `h5py.Group.require_dataset()`

    Returns
    -------
    dataset : h5py.Dataset
        Handle to the HDF5 dataset object that was written.
    """
    is_strlist = (isinstance(value, list)
        and all(isinstance(v, str) for v in value))
    if isinstance(value, str) or is_strlist:
        value = np.bytes_(value)
        # If user requested string then let numpy determine length, otherwise
        # throw error.
        if dtype != np.bytes_:
            raise NotImplementedError(f"can't convert string to {dtype}")
        dtype = value.dtype
        # h5py doesn't like empty strings?  Want to cover this case for
        # situations where we have to write out the name of some optional
        # input file, but the file wasn't provided.
        if dtype == np.dtype("S0"):
            dtype = np.dtype("S1")
    shape = () if np.isscalar(value) else value.shape
    dset = group.require_dataset(name, shape, dtype, data=value, **kwargs)
    dset.attrs["description"] = np.bytes_(description)
    if units is not None:
        dset.attrs["units"] = np.bytes_(units)
    return dset

def coswin(t, eta):
    """
    Continuous cosine window

    Parameters
    ----------
    t : Union[float, np.ndarray[float]]
        Time variable in interval [-0.5, 0.5]
    eta : float
        Pedestal height in interval [0, 1]

    Returns
    -------
    window : Union[float, np.ndarray[float]]
        Cosine window coefficient(s)
    """
    a = (1 + eta) / 2
    b = (1 - eta) / 2
    return a + b * np.cos(2 * np.pi * t)


def kaiser(t, beta):
    """
    Continuous Kaiser window

    Parameters
    ----------
    t : Union[float, np.ndarray[float]]
        Time variable in interval [-0.5, 0.5]
    beta : float
        Shape parameter in interval [0, inf]

    Returns
    -------
    window : Union[float, np.ndarray[float]]
        Kaiser window coefficient(s)
    """
    return np.i0(beta * np.sqrt(1 - 4 * t**2)) / np.i0(beta)


def quaternion_to_euler(t, q, orbit, ellipsoid=Ellipsoid()):
    """Convert rcs2xyz quaternion to rcs2tcn Euler sequence defined in
    JPL D-102264

    Parameters
    ----------
    t : float
        Time stamp of quaternion, seconds relative to orbit reference epoch
    q : isce3.core.Quaternion
        Quaternion representing rotation from RCS frame to ECEF XYZ frame.
    orbit : isce3.core.Orbit
        Orbit object
    ellipsoid : Optional[isce3.core.Ellipsoid]
        Ellipsoid used to determine geodetic nadir vector.  Default=WGS84

    Returns
    -------
    yaw, pitch, roll : float
        Euler angles in radians
    """
    rcs2xyz = q.to_rotation_matrix()
    pos, vel = orbit.interpolate(t)
    tcn2xyz = isce3.core.geodetic_tcn(pos, vel, ellipsoid).asarray()
    xyz2tcn = tcn2xyz.transpose()
    rcs2tcn = xyz2tcn @ rcs2xyz
    return Quaternion(rcs2tcn).to_ypr()


def require_lut_axes(group, epoch, t, r, kind):
    """Get lookup table axes or write them if they don't exist.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to read or write from
    epoch : isce3.core.DateTime
        Reference time stamp
    t : numpy.ndarray
        Time axis in seconds relative to `epoch` to write if "zeroDopplerTIme"
        key is not found in group.
    r : numpy.ndarray
        Range axis in meters to write if "slantRange" key is not found in group.
    kind : str
        Trailing part of dataset description.  Will write out a string like
        "... dimension corresponding to " + `kind`

    Returns
    -------
    t : numpy.ndarray
        Contents of group["zeroDopplerTime"] if found else the input `t`
    r : numpy.ndarray
        Contents of group["slantRange"] if found else the input `r`
    """
    name = "zeroDopplerTime"
    if name in group:
        t = group[name][:]
    else:
        write_dataset(group, name, np.float64, t,
            "Zero doppler time dimension corresponding to " + kind,
            time_units(epoch))

    name = "slantRange"
    if name in group:
        r = group[name][:]
    else:
        write_dataset(group, name, np.float64, r,
            "Slant range dimension corresponding to " + kind, "meters")

    return t, r


class SLC(h5py.File):
    def __init__(self, *args, band="LSAR", product="RSLC", **kw):
        super().__init__(*args, **kw)
        self.band = band
        self.product = product
        self.root = self.create_group(f"/science/{band}/{product}")
        self.idpath = f"/science/{band}/identification"
        self.attrs["Conventions"] = np.bytes_("CF-1.7")
        self.attrs["contact"] = np.bytes_("nisar-sds-ops@jpl.nasa.gov")
        self.attrs["institution"] = np.bytes_("NASA JPL")
        self.attrs["mission_name"] = np.bytes_("NISAR")
        self.attrs["reference_document"] = np.bytes_("D-102268 NISAR NASA SDS "
            "Product Specification Level-1 Range Doppler Single Look Complex "
            "L1_RSLC")
        self.attrs["title"] = np.bytes_("NISAR L1 RSLC Product")

    def create_dataset(self, *args, **kw):
        log.debug(f"Creating dataset {args[0]}")
        return super().create_dataset(*args, **kw)

    def create_group(self, *args, **kw):
        log.debug(f"Creating group {args[0]}")
        return super().create_group(*args, **kw)

    def _set_range_window(self, group: h5py.Group, window_name: str,
                          window_shape: float, sample_rate: float,
                          bandwidth: float) -> None:
        n = 256  # according to NISAR_PIX
        f = np.fft.fftfreq(n, 1.0 / sample_rate)
        # XXX What's the desired frequency order?  Assume sorted for nice plot
        f = np.fft.fftshift(f)
        mask = abs(f) <= (bandwidth / 2.0)
        values = np.zeros(n, np.float32)
        window_name = window_name.lower()
        window = coswin if window_name == "cosine" else kaiser
        if window_name not in ("kaiser", "cosine"):
            raise NotImplementedError("Only Kaiser or Cosine windows are "
                f"supported, got {window_name}")
        values[mask] = window(f[mask] / bandwidth, window_shape)

        dset_name = "rangeChirpWeighting"
        if dset_name not in group:
            dset = write_dataset(group, dset_name, np.float32, values,
                "1-D array in frequency domain for range processing. This is "
                "used for processing L0b to L1. FFT length=256 (assumed)",
                units="1")
            dset.attrs["window_name"] = np.bytes_(window_name)
            dset.attrs["window_shape"] = window_shape


    def set_algorithms(self, *, demInterpolation="bilinear",
                       rfiDetection="ST-EVD", rfiMitigation="ST-EVD",
                       rangeCompression="FFT convolution",
                       elevationAntennaPatternCorrection=True,
                       rangeSpreadingLossCorrection=True,
                       dopplerCentroidEstimation="geometric",
                       azimuthPresumming="BLU",
                       azimuthCompression="time-domain backprojection"):
        def bool_to_str(enabled: bool):
            assert isinstance(enabled, bool)
            return "enabled" if enabled else "disabled"
        g = self.root.require_group("metadata/processingInformation/algorithms")
        write_dataset(g, "demInterpolation", np.bytes_, demInterpolation,
            "DEM interpolation method")
        write_dataset(g, "rfiDetection", np.bytes_, rfiDetection,
            "Algorithm used for radio frequency interference (RFI) detection")
        write_dataset(g, "rfiMitigation", np.bytes_, rfiMitigation,
            'Algorithm used for radio frequency interference (RFI) mitigation, '
            'either "ST-EVD" or "FDNF" (or "disabled" if no RFI mitigation was '
            'applied)')
        write_dataset(g, "rangeCompression", np.bytes_, rangeCompression,
            "Algorithm for focusing the data in the range direction")
        write_dataset(g, "elevationAntennaPatternCorrection", np.bytes_,
            bool_to_str(elevationAntennaPatternCorrection),
            "Algorithm for calibrating the antenna pattern")
        write_dataset(g, "rangeSpreadingLossCorrection", np.bytes_,
            bool_to_str(rangeSpreadingLossCorrection),
            "Algorithm for calibrating range fading")
        write_dataset(g, "dopplerCentroidEstimation", np.bytes_,
            dopplerCentroidEstimation,
            "Algorithm for calculating Doppler centroid")
        write_dataset(g, "azimuthPresumming", np.bytes_, azimuthPresumming,
            "Algorithm for regridding and filling gaps in the raw data "
            "in azimuth")
        write_dataset(g, "azimuthCompression", np.bytes_, azimuthCompression,
            "Algorithm for focusing the data in the azimuth direction")
        write_dataset(g, "softwareVersion", np.bytes_, isce3.__version__,
            "Software version used for processing")


    def set_parameters(self, dop: LUT2d, epoch: DateTime, frequency='A',
                       range_window_name="Kaiser", range_window_shape=1.6,
                       range_oversample=1.2, azimuth_envelope=None,
                       runconfig_contents: str = ""):
        """Write processing parameters to the NISAR product.

        Parameters
        ----------
        dop : isce3.core.LUT2d
            Doppler frequency (Hz) vs azimuth time (s) and slant range (m).
        epoch : isce3.core.DateTime
            Reference epoch for azimuth time.
        frequency : str in {'A', 'B'}, optional
            Frequency sub-band
        range_window_name : str in {'Cosine', 'Kaiser'}, optional
            Name of range spectral window
        range_window_shape : float, optional
            Shape parameter for range spectral window.  Beta for Kaiser window,
            pedestal height for cosine window.
        range_oversample : float, optional
            Ratio of range sample rate to range bandwidth.
        azimuth_envelope : np.ndarray or None, optional
            Array containing azimuth spectral envelope.  If None then write ones
        runconfig_contents : str, optional
            Contents of run configuration YAML file.
        """
        log.info(f"Saving Doppler for frequency {frequency}")
        g = self.root.require_group("metadata/processingInformation/parameters")
        # Actual LUT goes into a subdirectory, not created by serialization.
        name = f"frequency{frequency}"
        fg = g.require_group(name)
        add_cal_layer(fg, dop, "dopplerCentroid", epoch, "hertz",
            f"2D LUT of Doppler centroid for frequency {frequency}")
        self._set_range_window(g, range_window_name, range_window_shape,
                               range_oversample, 1.0)

        if azimuth_envelope is None:
            azimuth_envelope = np.ones(256, np.float32)

        name = "azimuthChirpWeighting"
        if name not in g:
            write_dataset(g, name, np.float32, azimuth_envelope,
                "1-D array in frequency domain for azimuth processing. This is "
                "used for processing L0b to L1. FFT length=256 (assumed)",
                units="1")
        # TODO ref height
        if "referenceTerrainHeight" not in g:
            n = dop.data.shape[0]
            write_dataset(g, "referenceTerrainHeight", np.float32, np.zeros(n),
                "Reference Terrain Height as a function of time", "meters")

        t = dop.y_start + dop.y_spacing * np.arange(dop.data.shape[0])
        r = dop.x_start + dop.x_spacing * np.arange(dop.data.shape[1])
        require_lut_axes(g, epoch, t, r, "processing information records")

        write_dataset(g, "runConfigurationContents", np.bytes_,
            runconfig_contents, "Contents of the run configuration file "
            "with parameters used for processing")


    def swath(self, frequency="A") -> h5py.Group:
        return self.root.require_group(f"swaths/frequency{frequency}")

    def add_polarization(self, frequency="A", pol="HH"):
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
        pols = np.bytes_([pol])  # careful not to promote to unicode below
        g = self.swath(frequency)
        name = "listOfPolarizations"
        if name in g:
            old_pols = np.array(g[name])
            assert pols[0] not in old_pols
            pols = np.append(old_pols, pols)
            del g[name]
        dset = g.create_dataset(name, data=pols)
        desc = f'List of processed polarization layers with frequency {frequency}'
        dset.attrs["description"] = np.bytes_(desc)

    def create_image(self, frequency="A", pol="HH", **kw) -> h5py.Dataset:
        log.info(f"Creating SLC image for frequency={frequency} pol={pol}"
            f" with HDF5 options={kw}")
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
        self.add_polarization(frequency, pol)
        kw.setdefault("dtype", complex32)
        dset = self.swath(frequency).create_dataset(pol, **kw)
        dset.attrs["description"] = np.bytes_(f"Focused RSLC image ({pol})")
        dset.attrs["units"] = np.bytes_("1")
        return dset

    def update_swath(self, grid: RadarGridParameters, orbit: Orbit,
                     range_bandwidth: float, frequency: str,
                     azimuth_bandwidth: float, acquired_prf: float,
                     acquired_range_bandwidth: float, acquired_fc: float,
                     sub_swaths: np.ndarray):
        """Write swath metadata.

        Parameters
        ----------
        grid : isce3.product.RadarGridParameters
            Focused output grid parameters
        orbit : isce3.core.Orbit
            Radar orbit
        range_bandwidth : float
            Processed range bandwidth in Hz
        frequency : str in {"A", "B"}
            Frequency sub-band identifier
        azimuth_bandwidth : float
            Processed azimuth bandwidth in Hz
        acquired_prf : float
            Smallest PRF (in Hz) among input files.  May differ from `grid.prf`.
        acquired_range_bandwidth : float
            Largest chirp bandwidth (in Hz) among input files.
        acquired_fc : float
            Largest center frequency (in Hz) among input files.
        sub_swaths : numpy.ndarray
            Array of shape (nswaths, ntimes, 2) containing the valid
            [start, stop) grid indices of each sub swath.
        """
        t = grid.sensing_times
        r = grid.slant_ranges
        fc = isce3.core.speed_of_light / grid.wavelength
        epoch = grid.ref_epoch

        daz, dgr = get_radar_grid_nominal_ground_spacing(grid, orbit)

        g = self.swath(frequency)
        # Time scale is in parent of group.  Use require_dataset to assert
        # matching time scale on repeated calls.
        d = g.parent.require_dataset("zeroDopplerTime", t.shape, t.dtype, data=t)
        d.attrs["units"] = np.bytes_(time_units(epoch))
        d.attrs["description"] = np.bytes_(
            "CF compliant dimension associated with azimuth time")

        d = g.parent.require_dataset("zeroDopplerTimeSpacing", (), float)
        d[()] = t.spacing
        d.attrs["units"] = np.bytes_("seconds")
        d.attrs["description"] = np.bytes_("Time interval in the along-track "
            "direction for raster layers. This is same as the spacing between "
            "consecutive entries in the zeroDopplerTime array")

        d = g.require_dataset("slantRange", r.shape, r.dtype, data=r)
        d.attrs["units"] = np.bytes_("meters")
        d.attrs["description"] = np.bytes_("CF compliant dimension associated"
                                            " with slant range")

        d = g.require_dataset("slantRangeSpacing", (), float)
        d[()] = r.spacing
        d.attrs["units"] = np.bytes_("meters")
        d.attrs["description"] = np.bytes_("Slant range spacing of grid. Same"
            " as difference between consecutive samples in slantRange array")

        d = g.require_dataset("processedCenterFrequency", (), float)
        d[()] = fc
        d.attrs["units"] = np.bytes_("hertz")
        d.attrs["description"] = np.bytes_("Center frequency of the processed"
                                            " image in hertz")

        write_dataset(g, "acquiredCenterFrequency", float, acquired_fc,
            "Center frequency of the acquisition in hertz. In case of mode "
            "combination, this corresponds to the mode with highest center "
            "frequency.", "hertz")
        write_dataset(g, "acquiredRangeBandwidth", float,
            acquired_range_bandwidth, "Acquisition range bandwidth in "
            "hertz. In case of mode combination, this corresponds to mode with "
            "largest bandwidth.", "hertz")
        write_dataset(g, "nominalAcquisitionPRF", float, acquired_prf,
            "Nominal PRF of acquisition. In case of mode combination, this "
            "corresponds to mode with least nominal PRF.", "hertz")
        write_dataset(g, "processedAzimuthBandwidth", float, azimuth_bandwidth,
            "Processed azimuth bandwidth in hertz", "hertz")
        write_dataset(g, "processedRangeBandwidth", float, range_bandwidth,
            "Processed range bandwidth in hertz", "hertz")
        write_dataset(g, "sceneCenterAlongTrackSpacing", float, daz,
            "Nominal along-track spacing in meters between consecutive lines "
            "near mid swath of the RSLC image", "meters")
        write_dataset(g, "sceneCenterGroundRangeSpacing", float, dgr,
            "Nominal ground range spacing in meters between consecutive pixels "
            "near mid swath of the RSLC image", "meters")

        write_dataset(g, "numberOfSubSwaths", 'uint8', len(sub_swaths),
            "Number of swaths of continuous imagery, due to transmit gaps", "1")
        nth_names = ["1st", "2nd", "3rd"]
        for i, swath in enumerate(sub_swaths):
            name = f"validSamplesSubSwath{i + 1}"
            nth = f"{i + 1}th"
            if i < 3:
                nth = nth_names[i]
            write_dataset(g, name, 'uint32', swath,
                f"First and last valid sample in each line of {nth} subswath",
                "1")

    def set_orbit(self, orbit: Orbit, type="Custom"):
        log.info("Writing orbit to SLC")
        g = self.root.require_group("metadata/orbit")
        orbit.save_to_h5(g)
        # Add description attributes.  Should these go in saveToH5 method?
        g["time"].attrs["description"] = np.bytes_("Time vector record. This"
            " record contains the time corresponding to position and velocity"
            " records")
        g["position"].attrs["description"] = np.bytes_("Position vector"
            " record. This record contains the platform position data with"
            " respect to WGS84 G1762 reference frame")
        g["velocity"].attrs["description"] = np.bytes_("Velocity vector"
            " record. This record contains the platform velocity data with"
            " respect to WGS84 G1762 reference frame")
        g["interpMethod"].attrs["description"] = np.bytes_(
            'Orbit interpolation method, either "Hermite" or "Legendre"')
        # Orbit source/type
        g["orbitType"].attrs["description"] = np.bytes_(
            'Orbit product type, either "FOE", "NOE", "MOE", "POE", or'
            ' "Custom", where "FOE" stands for Forecast Orbit Ephemeris,'
            ' "NOE" is Near real-time Orbit Ephemeris, "MOE" is Medium'
            ' precision Orbit Ephemeris, and "POE" is Precise Orbit'
            ' Ephemeris')

    def set_attitude(self, attitude: Attitude, orbit: Orbit,
                     ellipsoid=Ellipsoid(), type="Custom"):
        log.info("Writing attitude to SLC")
        g = self.root.require_group("metadata/attitude")
        d = g.require_dataset("attitudeType", (), "S10", data=np.bytes_(type))
        d.attrs["description"] = np.bytes_(
            'Attitude type, either "FRP", "NRP", "PRP, or "Custom", where'
            ' "FRP" stands for Forecast Radar Pointing, "NRP" is Near'
            ' Real-time Pointing, and "PRP" is Precise Radar Pointing')
        t = np.asarray(attitude.time)
        d = g.require_dataset("time", t.shape, t.dtype, data=t)
        d.attrs["description"] = np.bytes_("Time vector record. This record"
            " contains the time corresponding to attitude and quaternion"
            " records")
        d.attrs["units"] = np.bytes_(time_units(attitude.reference_epoch))
        qv = np.array([[q.w, q.x, q.y, q.z] for q in attitude.quaternions])
        d = g.require_dataset("quaternions", qv.shape, qv.dtype, data=qv)
        d.attrs["units"] = np.bytes_("1")
        d.attrs["description"] = np.bytes_("Attitude quaternions"
                                            " (q0, q1, q2, q3)")
        ypr = np.rad2deg([quaternion_to_euler(ti, qi, orbit, ellipsoid)
            for (ti, qi) in zip(attitude.time, attitude.quaternions)])
        write_dataset(g, "eulerAngles", np.float64, ypr[:,::-1],
            "Attitude Euler angles (roll, pitch, yaw)", "degrees")

    def copy_identification(self, raw: Raw, *, track: int = 0, frame: int = 0,
                            absolute_orbit_number: Optional[int] = None,
                            polygon: Optional[str] = None,
                            start_time: Optional[DateTime] = None,
                            end_time: Optional[DateTime] = None,
                            mission_id: Optional[str] = None,
                            instrument_name: Optional[str] = None,
                            frequencies: Optional[str] = None,
                            planned_datatake_id: Optional[str] = None,
                            planned_observation_id: Optional[str] = None,
                            is_urgent: Optional[bool] = None,
                            product_spec_version: str = "1.1.2",
                            processing_center: str = "JPL",
                            granule_id: str = "None",
                            product_version: str = "0.1.0",
                            processing_type: str = "CUSTOM",
                            is_dithered: bool = False,
                            is_mixed_mode: bool = False):
        """
        Populate identification metadata with a combination of copied values
        from L0B and user data.

        always copied from L0B:
            diagnosticModeFlag
            isGeocoded
            lookDirection
            orbitPassDirection

        copied from L0B if associated input argument is None:
            absoluteOrbitNumber
            boundingPolygon
            instrumentName
            isUrgentObservation
            listOfFrequencies
            missionId
            plannedDatatakeId
            plannedObservationId
            zeroDopplerEndTime
            zeroDopplerStartTime

        always populated based on input values:
            frameNumber
            granuleId
            isDithered
            isMixedMode
            processingCenter
            processingType
            productSpecificationVersion
            productVersion
            trackNumber

        always populated independently:
            processingDateTime
            productLevel
            productType
            radarBand
        """
        log.info(f"Populating identification based on {raw.filename}")
        # Most parameters are just copies of input ID.
        if self.idpath in self.root:
            del self.root[self.idpath]
        with h5py.File(raw.filename, 'r', libver='latest', swmr=True) as fd:
            self.root.copy(fd[raw.IdentificationPath], self.idpath)
        g = self.root[self.idpath]
        # Delete units from diagnosticModeFlag if present (spec changed).
        name = "diagnosticModeFlag"
        if name in g and "units" in g[name].attrs:
            log.warning("Input L0B has undesired 'units' attribute on "
                "diagnosticModeFlag dataset.  Will omit from output RSLC.")
            del g[name].attrs["units"]
        # Of course product type is different.
        d = set_string(g, "productType", self.product)
        d.attrs["description"] = np.bytes_("Product type")
        # L0B doesn't know about track/frame, so have to add it.
        d = g.require_dataset("trackNumber", (), 'uint8', data=track)
        d.attrs["units"] = np.bytes_("1")
        d.attrs["description"] = np.bytes_("Track number")
        d = g.require_dataset("frameNumber", (), 'uint16', data=frame)
        d.attrs["units"] = np.bytes_("1")
        d.attrs["description"] = np.bytes_("Frame number")
        # Polygon different due to reskew and possibly multiple input L0Bs.
        if polygon is not None:
            d = set_string(g, "boundingPolygon", polygon)
            d.attrs["epsg"] = 4326
            d.attrs["description"] = np.bytes_(descriptions.bounding_polygon)
            d.attrs["ogr_geometry"] = np.bytes_("polygon")
        else:
            log.warning("SLC bounding polygon not updated.  Using L0B polygon.")
        # Start/end time can be customized via runconfig and generally are
        # different anyway due to reskew.
        if start_time is not None:
            d = set_string(g, "zeroDopplerStartTime", start_time.isoformat())
            d.attrs["description"] = np.bytes_(
                "Azimuth start time of the product")
        else:
            log.warning("SLC start time not updated.  Using L0B start time.")
        if end_time is not None:
            d = set_string(g, "zeroDopplerEndTime", end_time.isoformat())
            d.attrs["description"] = np.bytes_(
                "Azimuth stop time of the product")
        else:
            log.warning("SLC end time not updated.  Using L0B end time.")

        # It should be pretty safe to copy mission_id and instrument_name from
        # the input L0B, so don't bother warning if we do that.
        if mission_id is not None:
            d = set_string(g, "missionId", mission_id)
            d.attrs["description"] = np.bytes_("Mission identifier")
            log.info(f"Updating missionId to {mission_id}")

        # Add "LSAR" instrument name if it wasn't in either L0B or arg list.
        if "instrumentName" not in g and instrument_name is None:
            instrument_name = self.band

        if instrument_name is not None:
            d = set_string(g, "instrumentName", instrument_name)
            d.attrs["description"] = np.bytes_("Name of the instrument used "
                "to collect the remote sensing data provided in this product")
            log.info(f"Updating instrumentName to {instrument_name}")

        if absolute_orbit_number is not None:
            d = g.require_dataset("absoluteOrbitNumber", (), np.uint32)
            d[()] = np.uint32(absolute_orbit_number)
            d.attrs["description"] = np.bytes_("Absolute orbit number")

        def set_string_list(group, key, values, desc):
            if key in group:
                # delete since we can't guarantee old list has same length
                del group[key]
            d = group.create_dataset(key, data=np.bytes_(values))
            d.attrs["description"] = np.bytes_(desc)

        if frequencies is not None:
            set_string_list(g, "listOfFrequencies", frequencies,
                "List of frequency layers available in the product")

        if planned_datatake_id is not None:
            set_string_list(g, "plannedDatatakeId", planned_datatake_id,
                "List of planned datatakes included in the product")

        if planned_observation_id is not None:
            set_string_list(g, "plannedObservationId", planned_observation_id,
                "List of planned observations included in the product")

        if is_urgent is not None:
            d = set_string(g, "isUrgentObservation", str(is_urgent))
            d.attrs["description"] = np.bytes_(
                'Flag indicating if observation is nominal ("False") '
                'or urgent ("True")')

        d = set_string(g, "productSpecificationVersion", product_spec_version)
        d.attrs["description"] = np.bytes_("Product specification version "
            "which represents the schema of this product")

        d = set_string(g, "processingCenter", processing_center)
        d.attrs["description"] = np.bytes_("Data processing center")

        d = set_string(g, "granuleId", granule_id)
        d.attrs["description"] = np.bytes_(
            "Unique granule identification name")

        d = set_string(g, "productVersion", product_version)
        d.attrs["description"]= np.bytes_("Product version which represents "
            "the structure of the product and the science content governed by "
            "the algorithm, input data, and processing parameters")

        d = set_string(g, "productLevel", "L1")
        d.attrs["description"] = np.bytes_("Product level. L0A: Unprocessed "
            "instrument data; L0B: Reformatted, unprocessed instrument data; "
            "L1: Processed instrument data in radar coordinates system; and "
            "L2: Processed instrument data in geocoded coordinates system")

        d = set_string(g, "radarBand", self.band[0])
        d.attrs["description"] = np.bytes_('Acquired frequency band, '
            'either "L" or "S"')

        d = set_string(g, "processingType", processing_type)
        d.attrs["description"] = np.bytes_(
            "Nominal (or) Urgent (or) Custom (or) Undefined")

        d = set_string(g, "isDithered", str(is_dithered))
        d.attrs["description"] = np.bytes_('"True" if the pulse timing was '
            'varied (dithered) during acquisition, "False" otherwise.')

        d = set_string(g, "isMixedMode", str(is_mixed_mode))
        d.attrs["description"] = np.bytes_('"True" if this product is a '
            'composite of data collected in multiple radar modes, '
            '"False" otherwise.')

        # only report to integer seconds
        now = datetime.now(timezone.utc).isoformat()[:19]
        d = set_string(g, "processingDateTime", now)
        d.attrs["description"] = np.bytes_("Processing UTC date and time in "
            "the format YYYY-mm-ddTHH:MM:SS")


    def set_geolocation_grid(self, orbit: Orbit, grid: RadarGridParameters,
                             doppler: LUT2d, epsg=4326, dem=DEMInterpolator(),
                             **kw):
        log.info(f"Creating geolocationGrid.")
        # TODO Get DEM stats.  Until then just span all Earthly values.
        heights = np.linspace(-500, 9000, 20)
        # Figure out decimation factors that give < 500 m spacing.
        max_spacing = 500.
        t = (grid.sensing_mid +
             (grid.ref_epoch - orbit.reference_epoch).total_seconds())
        _, v = orbit.interpolate(t)
        dx = np.linalg.norm(v) / grid.prf

        group_name = f"{self.root.name}/metadata/geolocationGrid"
        rslc_doppler = LUT2d()  # RSLCs are zero-Doppler by definition

        # Create a new geolocation radar grid with 5 extra points
        # before and after the starting and ending
        # zeroDopplerTime and slantRange
        extra_points = 5

        # Total number of samples along the azimuth and slant range
        # using around 500m sampling interval
        ysize = int(np.ceil(grid.length / (max_spacing / dx)))
        xsize = int(np.ceil(grid.width / \
            (max_spacing / grid.range_pixel_spacing)))

        # New geolocation grid
        geolocation_radargrid = \
            grid.resize_and_keep_startstop(ysize, xsize)
        geolocation_radargrid = \
            geolocation_radargrid.add_margin(extra_points,
                                             extra_points)

        # TODO Fix keyword args
        add_geolocation_grid_cubes_to_hdf5(self, group_name,
                                           geolocation_radargrid,
                                           heights,orbit, doppler,
                                           rslc_doppler, epsg)

    def write_stats(self, frequency, pol, stats):
        h5_ds = self.swath(frequency)[pol]
        write_stats_complex_data(h5_ds, stats)


    def add_calibration_section(self, frequency, pol,
                                az_time_orig_vect: np.array,
                                epoch: DateTime,
                                slant_range_orig_vect: np.array):
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"

        # TODO agree on LUT postings.
        calibration_section_sampling = 50
        t = az_time_orig_vect[::calibration_section_sampling]
        r = slant_range_orig_vect[::calibration_section_sampling]

        cal_group = self.root.require_group("metadata/calibrationInformation")

        # TODO Populate backscatter conversion layers.  Plan is for beta0=1,
        # but the others should account for the ellipsoid incidence angle.
        geo_group = cal_group.require_group("geometry")

        # NOTE Spec changed so that now each subgroup has its own LUT2d axes.
        t, r = require_lut_axes(geo_group, epoch, t, r, "calibration records")

        dummy_array = np.ones((t.size, r.size), dtype=np.float32)

        if "beta0" not in geo_group:
            write_dataset(geo_group, "beta0", np.float32, dummy_array,
                "2D LUT to convert DN to beta 0 assuming as a function"
                " of zero doppler time and slant range", "1")

        if "sigma0" not in geo_group:
            write_dataset(geo_group, "sigma0", np.float32, dummy_array,
                "2D LUT to convert DN to sigma 0 assuming as a function"
                " of zero doppler time and slant range", "1")

        if "gamma0" not in geo_group:
            write_dataset(geo_group, "gamma0", np.float32, dummy_array,
                "2D LUT to convert DN to gamma 0 as a function of zero"
                " doppler time and slant range", "1")

        # TODO Populate EAP with real values.  Reporting unit gain for now.
        eap_group = cal_group.require_group(
            f"frequency{frequency}/elevationAntennaPattern")

        t, r = require_lut_axes(eap_group, epoch, t, r,
            "calibration elevationAntennaPattern records")

        dummy_array = np.ones((t.size, r.size), dtype=np.complex64)

        write_dataset(eap_group, pol, np.complex64, dummy_array,
            "Complex two-way elevation antenna pattern", "1")

        # TODO Populate NESZ with real values.  Using zero for now.
        nes0_group = cal_group.require_group(f"frequency{frequency}/nes0")
        t, r = require_lut_axes(nes0_group, epoch, t, r,
            "calibration nes0 records")

        write_dataset(nes0_group, pol, np.float32, np.zeros((t.size, r.size)),
            "Noise equivalent sigma zero", "1")

        # TODO Populate crosstalk with real values.  Reporting zero for now.
        xt_group = cal_group.require_group("crosstalk")

        name = "slantRange"
        if name in xt_group:
            r = xt_group[name][:]
        else:
            write_dataset(xt_group, name, np.float64, r,
                "Slant range dimension corresponding to crosstalk records",
                "meters")

        zero = np.zeros(len(r), np.complex64)

        write_dataset(xt_group, "txHorizontalCrosspol", np.complex64, zero,
            "Crosstalk in H-transmit channel expressed as ratio txV / txH", "1")
        write_dataset(xt_group, "txVerticalCrosspol", np.complex64, zero,
            "Crosstalk in V-transmit channel expressed as ratio txH / txV", "1")
        write_dataset(xt_group, "rxHorizontalCrosspol", np.complex64, zero,
            "Crosstalk in H-receive channel expressed as ratio rxV / rxH", "1")
        write_dataset(xt_group, "rxVerticalCrosspol", np.complex64, zero,
            "Crosstalk in V-receive channel expressed as ratio rxH / rxV", "1")


    def set_rfi_results(self, rfi_results):
        """
        Store the results of radio frequency interference (RFI) analysis

        Parameters
        ----------
        rfi_results: dict[tuple[str, str], list[tuple[float, int]]]
            Results of RFI analysis organized as a dict keyed by
            (frequency, polarization) pairs where each value is a list of
            (rfi_likelihood, num_pulses) pairs, one for each observation.
            For example, {("A", "HH"): [(0.05, 1000)]} would be a valid
            argument.
        """
        g = self.root.require_group("metadata/calibrationInformation")
        for (frequency, pol), rfi_likelihoods in rfi_results.items():
            num_pulses = sum(n for (_, n) in rfi_likelihoods)
            weighted_sum = sum(x * n for (x, n) in rfi_likelihoods)
            average_rfi_likelihood = weighted_sum / num_pulses
            d = g.require_dataset(f"frequency{frequency}/{pol}/rfiLikelihood",
                shape=(), dtype=np.float64, data=average_rfi_likelihood)
            d.attrs["description"] = np.bytes_(
                "Severity of radio frequency interference (RFI) contamination "
                "in the data. Value is in the interval [0,1], where 0: lowest "
                "severity, and 1: highest severity (or NaN if RFI detection "
                "was skipped)")
            d.attrs["units"] = np.bytes_("1")

    def set_inputs(self, *, l0bGranules=[""], orbitFiles=[""],
                   attitudeFiles=[""], auxcalFiles=[""], configFiles=[""],
                   demSource=""):
        g = self.root.require_group("metadata/processingInformation/inputs")
        write_dataset(g, "l0bGranules", np.bytes_, l0bGranules,
            "List of input L0B products used")
        write_dataset(g, "orbitFiles", np.bytes_, orbitFiles,
            "List of input orbit files used")
        write_dataset(g, "attitudeFiles", np.bytes_, attitudeFiles,
            "List of input attitude files used")
        write_dataset(g, "auxcalFiles", np.bytes_, auxcalFiles,
            "List of input calibration files used")
        write_dataset(g, "configFiles", np.bytes_, configFiles,
            "List of input config files used")
        write_dataset(g, "demSource", np.bytes_, demSource,
            "Description of the input digital elevation model (DEM)")

    def set_calibration(self, cal: RslcCalibration, frequency: str) -> None:
        name = f"metadata/calibrationInformation/frequency{frequency}"
        g = self.root.require_group(name)
        write_dataset(g, "commonDelay", np.float64, cal.common_delay,
            "Range delay correction applied to all polarimetric channels",
            "meters")
        write_dataset(g, "faradayRotation", np.float64, 0.0,
            "Faraday rotation correction applied in processing", "radians")
        # Always write out parameters for all polarizations.
        for pol in ("HH", "HV", "VH", "VV"):
            pg = g.require_group(pol)
            chan = getattr(cal, pol.lower())
            amp = np.abs(chan.scale)
            phs = np.angle(chan.scale)
            write_dataset(pg, "differentialDelay", np.float64, chan.delay,
                f"Range delay correction applied to {pol} channel", "meters")
            write_dataset(pg, "differentialPhase", np.float64, phs,
                f"Phase correction applied to {pol} channel", "radians")
            write_dataset(pg, "scaleFactor", np.float64, amp,
                f"Scale factor applied to {pol} channel complex amplitude "
                "(at antenna boresite)", "1")
            write_dataset(pg, "scaleFactorSlope", np.float64, chan.scale_slope,
                f"Slope of scale factor applied to {pol} channel complex "
                "amplitude with respect to elevation angle", "radians^-1")
