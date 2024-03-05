"""
Some functionalities for corner reflectors from RSLC product
"""
from dataclasses import dataclass
import typing
import numpy as np
import warnings

from isce3.core import Ellipsoid, LUT2d, LookSide
from isce3.geometry import geo2rdr, rdr2geo, DEMInterpolator, heading
from isce3.cal.point_target_info import get_chip, oversample
from isce3.antenna import geo2ant, ant2geo

# datatype aliases
VectorFloat = typing.Union[typing.Sequence[float], np.ndarray]

# classes


@dataclass(frozen=True)
class CRInfoSlc:
    """Corner reflector info extracted/estimated from SLC.

    Attributes
    ----------
    amp_pol : dict of {str: complex}
        Complex amplitude of all products from RSLC.
        The items of the dict has a format of {TxRx_Pol:Amplitude}.
    llh : sequence or array of three floats
        Geodetic longitude, latitude and height in (rad, rad, m)
    el_ant : float
        Elevation (EL) angle (rad) in antenna frame
    az_ant : float, default=0
        Azimuth (AZ) angle (rad) in antenna frame

    """
    amp_pol: typing.Dict[str, complex]
    llh: VectorFloat
    el_ant: float
    az_ant: float = 0.0


class OutOfSlcBoundWarning(UserWarning):
    """Warning when CR LLH is out of SLC azimuth-range bound"""
    pass


# functions
def est_peak_loc_cr_from_slc(slc, cr_llh, *, freq_band='A',
                             ovs_fact=64, num_pixels=16, rel_pow_th_db=0.0,
                             rg_tol=0.1, max_iter=60, doppler=LUT2d(),
                             ellipsoid=Ellipsoid()):
    """
    Estimate exact peak value and location of corner reflector(s) (CR)
    from RSLC product based on approximate LLH for a desired frequency
    band and over all TxRx polarizations, either linear or circular basis.

    Parameters
    ----------
    slc : nisar.products.readers.SLC
        RSLC reader object for quad-pol data over CR.
    cr_llh : 1-D/2-D sequence or array of three floats
        Approximate Geodetic Longitude, latitude, height in (rad, rad, m) of
        CR(s). For more than one CR, the 2-D array shall have shape `Nx3`
        where  `N` is number of CRs.
    freq_band : {'A', 'B'}, default='A'
        Frequency band char of the RSLC product
    ovs_fact : int, default=64
        Oversampling factor used in interpolating a 2-D box/chip around
        `cr_llh` with total `num_pixels` in each direction.
    num_pixels : int, default=16
        Number of pixels used in interpolation in both range and azimuth.
    rel_pow_th_db : float, default=0.0
        Relative power threshold below the estimated peak value in (dB) to get
        a region around the peak location in range and azimuth and then compute
        coherently-averaged complex value within that region as peak value.
        If zero, the peak value will simply be an estimated peak value,
        otherwise, it will be a result of 2-D averaging of complex values
        within that relative threshold around the peak.
        The value must be non-negative!
    rg_tol : float, default=0.1
        Slant range (m) threshold in `rdr2geo` and step size in range gradient
        of `geo2rdr`.
    max_iter : int, default=60
        Max iterations used in `rdr2geo` and `geo2rdr`.
    doppler : isce3.core.LUT2d, default=zero Doppler
        Doppler in (Hz) as a function azimuth time and slant range in
        the form 2-D LUT used for RSLC. The default assumes zero-Doppler
        geometry for RSLC product radar grid where CRs are located and
        resampled.
    ellipsoid : isce3.core.Ellipsoid, default=WGS84
        Ellipsoidal model for the spheroid planet.

    Returns
    -------
    List of CRInfoSlc

    Raises
    ------
    ValueError
        For bad or out of range frequency or polarizations
    RuntimError
        For zero slant range defined from orbit to corner reflector
        Compact Pol or missing co-pol products.

    Warnings
    --------
    OutOfSlcBoundWarning
        For "cr_llh" out of RSLC data margins within +/- half
        of `num_pixels`.
        If any CR is outside the bounds of the RSLC radar grid, they
        will be skipped and this warning will be emitted.

    """
    # check margin value
    if rel_pow_th_db < 0:
        raise ValueError('"rel_pow_th_db" must be non-negative!')

    # check the frequency band
    list_freq = slc.frequencies
    if freq_band not in list_freq:
        raise ValueError(f'Frequency band is out of range {list_freq}')

    # check and get a list of valid co-pol TxRx polarizations where Tx==Rx
    co_pols = [p for p in slc.polarizations[freq_band] if p[0] == p[1]]
    if len(co_pols) == 0:
        raise RuntimeError(
            'Tx and Rx Pols are not the same basis or missing Co-pol data!')
    # get list of cx-pol products if any
    cx_pols = [p for p in slc.polarizations[freq_band] if p[0] != p[1]]

    # generate 2-D array (Nx3) of LLH in (rad,rad,m) for all CRs
    llh_all = np.asarray(cr_llh)
    # expand the first axis if there is simply one CR
    if llh_all.ndim == 1:
        llh_all = llh_all[np.newaxis, :]
    # some common objects used for all CRs
    radgrid = slc.getRadarGrid(freq_band)
    orbit = slc.getOrbit()
    attitude = slc.getAttitude()
    # some common scalars
    n_pix_half = num_pixels // 2
    inv_ovs_fact = 1. / ovs_fact

    # loop over CRs
    cr_info = []
    for llh in llh_all:
        # get approx slant_range/pixel and azimuth_time/line of a CR
        azt, sr = geo2rdr(
            llh, ellipsoid, orbit, doppler, radgrid.wavelength,
            radgrid.lookside, maxiter=max_iter, delta_range=rg_tol
        )
        azb = (azt - radgrid.sensing_start) * radgrid.prf
        rgb = (sr - radgrid.starting_range) / radgrid.range_pixel_spacing

        if (azb < n_pix_half or azb > (radgrid.length - n_pix_half) or
                rgb < n_pix_half or rgb > (radgrid.width - n_pix_half)):
            warnings.warn(
                f'SLC (rg_pixel, az_pixel)=({rgb:.1f}, {azb:.1f}) for CR'
                f' of (lon, lat, hgt)={llh} (rad, rad, m)',
                category=OutOfSlcBoundWarning, stacklevel=2)
            continue
        # loop over polarizations per CR
        amp_cr = dict()
        azb_pol = []
        rgb_pol = []
        for pol in co_pols:
            # get decoded RSLC dataset per pol
            dset_slc = slc.getSlcDatasetAsNativeComplex(freq_band, pol)
            # get exact peak location and its complex value via freq-domain
            # oversampling around approximate range/azimuth bins
            # get a chip around the (azb, rgb) and interpolate.
            azb_chp_first, rgb_chp_first, dset_chp = get_chip(
                dset_slc, azb, rgb, nchip=num_pixels
            )
            chp_ovs = oversample(
                dset_chp, ovs_fact, return_slopes=False, baseband=False
            )
            azb_pk_ovs, rgb_pk_ovs = np.unravel_index(
                abs(chp_ovs).argmax(), chp_ovs.shape
            )
            # interpolated complex peak value and locations of a CR.
            # the coherently-averaged amp around the peak within around 3dB
            # is used to represent the complex peak value.
            # Note that simly co-pol data is used for averaged location of
            # the CR for all polarizations.
            amp_cr[pol], az_rg_slices = _amp_avg_around_peak(
                chp_ovs, (azb_pk_ovs, rgb_pk_ovs), rel_pow_th_db)
            azb_pol.append(azb_chp_first + azb_pk_ovs * inv_ovs_fact)
            rgb_pol.append(rgb_chp_first + rgb_pk_ovs * inv_ovs_fact)

            # get cx-pol data with the same Rx pol as that of co-pol
            # Note that RX side is the main driver in appeared location
            # of CR in RSLC for each polarization due to its narrowband
            # and longer group delay than TX side. That is why the
            # products with the same RX polarizations are used together.
            x_pol = [p for p in cx_pols if p[1] == pol[1]]
            if len(x_pol) == 1:
                x_pol = x_pol[0]
                # get decoded RSLC dataset per pol
                dset_slc_cx = slc.getSlcDatasetAsNativeComplex(freq_band, x_pol)
                # oversampling around approximate range/azimuth bins
                # get a chip around the (azb, rgb) of the peak of respective
                # co-pol and interpolate.
                _, _, dset_chp_cx = get_chip(
                    dset_slc_cx, azb, rgb, nchip=num_pixels
                )
                chp_ovs_cx = oversample(
                    dset_chp_cx, ovs_fact, return_slopes=False, baseband=False
                )
                # get averaged complex amplitude defined by co-pol
                # peak loc or 3-dB boundary
                amp_cr[x_pol] = np.mean(
                    chp_ovs_cx[az_rg_slices]
                )

        # use the averaged (azb, rgb) over all pols
        azb_cr = np.mean(azb_pol)
        rgb_cr = np.mean(rgb_pol)
        # get exact slant range and azimuth time of a CR
        azt_cr = radgrid.sensing_start + azb_cr / radgrid.prf
        srg_cr = radgrid.starting_range + rgb_cr * radgrid.range_pixel_spacing
        # get updated LLH of CR based on SLC
        # Note that it is assumed local height is pretty constant around each
        # CR site. Thus, a fixed-height DEM object w/o topography is locally
        # formed for each CR to be used in rdr2geo.
        dem = DEMInterpolator(llh[-1])
        llh_cr = rdr2geo(
            azt_cr, srg_cr, orbit, radgrid.lookside, 0.0, radgrid.wavelength,
            dem, threshold=rg_tol, maxiter=max_iter
        )
        # get S/C position ECEF and attitude quaternions
        # at interpolated azimuth time of a CR
        pos_sc, _ = orbit.interpolate(azt_cr)
        quat = attitude.interpolate(azt_cr)
        # estimate (EL, AZ) angles of a CR in antenna frame
        el_ant_cr, az_ant_cr = geo2ant(llh_cr, pos_sc, quat)
        # store estimated CR Info
        cr_info.append(
            CRInfoSlc(amp_cr, llh_cr, el_ant_cr, az_ant_cr)
        )
    return cr_info


def est_cr_az_mid_swath_from_slc(slc):
    """
    Estimate approximately optimum azimuth (AZ) angle for a CR at mid swath
    from RSLC product.

    Parameters
    ----------
    slc : nisar.products.readers.SLC
        The input RSLC product.

    Returns
    -------
    float
        Azimuth angle (radians) for a CR at mid swath within [0, 2*pi].
        This is the heading angle of the corner reflector, or the angle that
        the corner reflector boresight makes w.r.t geographic East, measured
        clockwise positive in the E-N plane.
        Equivalently, this is the heading of the base of the CR w.r.t
        North in the clockwise direction.

    Notes
    -----
    Mid swath is approximately defined at mechanical boresight angle by
    ignoring electrical squint and DEM!

    """
    # get mid azimuth time of slc
    rdr_grid = slc.getRadarGrid()
    az_time = rdr_grid.sensing_mid

    # get pos/vel at mid slc time
    orbit = slc.getOrbit()
    pos, vel = orbit.interpolate(az_time)

    # get quaternions at mid slc time
    attitude = slc.getAttitude()
    quat = attitude.interpolate(az_time)

    # Mechanical boresight is defined by (EL, AZ) = (0, 0) in antenna frame!
    llh_mid, _ = ant2geo(0, 0, pos, quat)
    head_mid = heading(*llh_mid[:2], vel)

    # adjust the heading per antenna look side
    # if right looking (-1) add a pi
    if rdr_grid.lookside == LookSide.Right:
        head_mid += np.pi
    # make sure value is within [0, 2*pi] to be consistent with CR AZ
    # definition.
    # This step is not really necessary!
    if head_mid < 0:
        head_mid += (2 * np.pi)

    return head_mid


def _amp_avg_around_peak(chp_ovs, azb_rgb_pk_loc, dynamic_range_db=3.0):
    """
    Calculate averaged complex amplitude around the peak location within
    desired relative dynamic range.

    Parameters
    ----------
    chp_ovs : np.ndarray(complex)
        2-D oversampled chip around the peak with shape (azimuth, range).
    azb_rgb_pk_loc : Tuple[int, int]
        (Azimuth bin, Range bin) of the peak location.
    dynamic_range_db : float, default=3.0
        Relative dynamic range in (dB) within which averaged peak
        value is calculated. A positive value!

    Returns
    -------
    complex
        Complex averaged amplitude around the peak.
    tuple[slice, slice]
        Azimuth and range bin slices

    """
    azb, rgb = azb_rgb_pk_loc
    pk_mag = abs(chp_ovs[azb, rgb])
    edge_mag = (10 ** (- dynamic_range_db / 20)) * pk_mag

    edge_slice_rg = _loc_left_right_val(abs(chp_ovs[azb]), rgb, edge_mag)
    edge_slice_az = _loc_left_right_val(abs(chp_ovs[:, rgb]), azb, edge_mag)

    amp_avg = chp_ovs[edge_slice_az, edge_slice_rg].mean()

    return amp_avg, (edge_slice_az, edge_slice_rg)


def _loc_left_right_val(arr: np.ndarray, index: int, val: float
                        ) -> slice:
    """Locate left/right indices of a value around an index in a 1-D array."""
    idx_right = index + np.where(arr[index:] <= val)[0][0]
    idx_left = index - np.where(arr[:index+1][::-1] <= val)[0][0]
    return slice(idx_left, idx_right + 1)
