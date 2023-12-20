#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import traceback
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

import isce3
import nisar


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, isce3.core.DateTime):
            return obj.isoformat()
        return super().default(obj)


CornerReflectorIterable = Union[
    Iterable[isce3.cal.TriangularTrihedralCornerReflector],
    Iterable[nisar.cal.CornerReflector],
]


def estimate_abscal_factor(
    corner_reflectors: CornerReflectorIterable,
    rslc: nisar.products.readers.SLC,
    freq: Optional[str] = None,
    pol: Optional[str] = None,
    external_orbit: Optional[isce3.core.Orbit] = None,
    *,
    nchip: int = 64,
    upsample_factor: int = 32,
    peak_find_domain: str = "time",
    nfit: int = 5,
    power_method: str = "box",
    pthresh: float = 3.0,
) -> List[dict[str, Any]]:
    r"""
    Estimate the absolute radiometric calibration factor of an RSLC product with one or
    more corner reflectors (CRs).

    Measures the average absolute radiometric calibration error for each CR in the scene,
    given the targets' known geodetic positions, orientations, and side lengths.
    For each corner reflector, the apparent radar cross-section (RCS) of the target is
    measured from the RSLC image data and compared with the predicted RCS based on the
    scene geometry and the dimensions of the trihedral. The output represents the ratio
    of measured RCS to predicted RCS for each input CR found in the scene.

    The processing assumes that all internal radiometric calibration (e.g. corrections
    for the antenna pattern and processor gain) and relative polarimetric calibration
    (e.g. corrections for channel imbalance, cross-talk, and Faraday rotation) have been
    applied to the RSLC data and that the data is normalized such that its intensity
    represents :math:`\beta_0` backscatter (radar brightness)\ [1]_.

    If the processing fails for any input CR, the error message and traceback info will
    be logged and the CR will be omitted from the results.

    Parameters
    ----------
    corner_reflectors : iterable
        Iterable of corner reflectors in the scene. The elements may be instances of
        `isce3.cal.TriangularTrihedralCornerReflector` or `nisar.cal.CornerReflector`.
        In the latter case, additional information about the survey date and plate
        motion velocity of each corner reflector is populated in the output.
    rslc : nisar.products.readers.SLC
        The input RSLC product.
    freq : {'A', 'B'} or None, optional
        The frequency sub-band of the data. Defaults to the science band in the RSLC
        product ('A' if available, otherwise 'B').
    pol : {'HH', 'VV', 'HV', 'VH'} or None, optional
        The transmit and receive polarization of the data. Defaults to the first
        co-polarization channel found in the specified band ('HH' if available,
        otherwise 'VV').
    external_orbit : isce3.core.Orbit or None, optional
        An optional external orbit dataset. If not specified, fall back to using the
        orbit data contained within the RSLC product. Defaults to None.
    nchip : int, optional
        The width, in pixels, of the square block of image data to extract centered
        around the target position for oversampling and peak finding. Must be >= 1.
        Defaults to 64.
    upsample_factor : int, optional
        The upsampling ratio. Must be >= 1. Defaults to 32.
    peak_find_domain : {'time', 'freq'}, optional
        Option controlling how the target peak position is estimated.

        'time':
          The default mode. The peak location is found in the time domain by detecting
          the maximum value within a square block of image data around the expected
          target location. The signal data is upsampled to improve precision.

        'freq':
          The peak location is found by estimating the phase ramp in the frequency
          domain. This mode is useful when target is well-focused, has high SNR, and is
          the only target in the neighborhood (often the case in point target
          simulations).
    nfit : int, optional
        The width, in *oversampled* pixels, of the square sub-block of image data to
        extract centered around the target position for fitting a quadratic polynomial
        to the peak. Note that this is the size in pixels *after upsampling*.
        Must be >= 3. Defaults to 5.
    power_method : {'box', 'integrated'}, optional
        The method for estimating the target signal power.

        'box':
          The default mode. Measures power using the rectangular box method, which
          assumes that the target response can be approximated by a 2-D rectangular
          function. The total power is estimated by multiplying the peak power by the
          3dB response widths in along-track and cross-track directions.

        'integrated':
          Measures power using the integrated power method. The total power is measured
          by summing the power of bins whose power exceeds a predefined minimum power
          threshold.
    pthresh : float, optional
        The minimum power threshold, measured in dB below the peak power, for estimating
        the target signal power using the integrated power method. This parameter is
        ignored if `power_method` is not 'integrated'. Defaults to 3.

    Returns
    -------
    results : list of dict
        A list of dicts containing one entry per corner reflector found in the scene.
        The dict of results for each corner reflector consists of the following keys:

        'id':
          The unique identifier of the corner reflector.

        'absolute_calibration_factor':
          The absolute radiometric calibration error for the corner reflector (the ratio
          of the measured RCS to the predicted RCS), in linear units.

        'timestamp':
          The corner reflector observation time.

        'frequency':
          The frequency sub-band of the data.

        'polarization':
          The transmit and receive polarization of the data.

        If the input corner reflectors were instances of `nisar.cal.CornerReflector`,
        the following additional keys are also populated:

        'survey_date':
          The date (and time) when the corner reflector was surveyed most recently prior
          to the radar observation.

        'velocity':
          The corner reflector velocity due to tectonic plate motion, as an
          East-North-Up (ENU) vector in meters per second (m/s). The velocity components
          are provided in local ENU coordinates with respect to the WGS 84 reference
          ellipsoid.

    Notes
    -----
    No corrections to the corner reflector position are applied for tectonic plate
    motion, solid earth tides, etc.

    References
    ----------
    .. [1] R. K. Raney, T. Freeman, R. W. Hawkins, and R. Bamler, “A plea for radar
       brightness,” Proceedings of IGARSS '94 - 1994 IEEE International Geoscience and
       Remote Sensing Symposium.
    """
    # Get frequency sub-band.
    if freq is None:
        freq = "A" if ("A" in rslc.frequencies) else "B"
    else:
        if freq not in {"A", "B"}:
            raise ValueError(f"freq must be 'A' or 'B' (or None), got {freq!r}")
        if freq not in rslc.frequencies:
            raise ValueError(
                f"freq {freq!r} not found in RSLC product. Available frequencies are"
                f" {set(rslc.frequencies)}"
            )

    # Get TxRx polarization.
    available_pols = rslc.polarizations[freq]
    if pol is None:
        pol = "HH" if ("HH" in available_pols) else "VV"
    else:
        if pol not in {"HH", "VV", "HV", "VH"}:
            raise ValueError(
                f"pol must be in {'HH', 'VV', 'HV', 'VH'} (or None), got {pol!r}"
            )
        if pol not in available_pols:
            raise ValueError(
                f"pol {pol!r} not found in freq {freq!r} of RSLC product. Available"
                f" polarizations are {set(available_pols)}"
            )

    # Get the RSLC image data for the specified frequency sub-band & polarization and
    # wrap it in a decoder layer that handles converting half-precision complex values
    # to single-precision.
    img_dataset = rslc.getSlcDataset(freq, pol)
    img_data = isce3.core.types.ComplexFloat16Decoder(img_dataset)

    # Get the radar grid on which the image data is sampled.
    radar_grid = rslc.getRadarGrid(freq)

    # Get the native Doppler centroid of the echo data and the Doppler of the image grid
    # that the focused data was projected onto (always zero Doppler for NISAR
    # radar-domain products).
    native_doppler = rslc.getDopplerCentroid(freq)
    image_grid_doppler = isce3.core.LUT2d()

    # Reference ellipsoid is assumed to be WGS 84 for corner reflector data and for
    # all NISAR processing.
    ellipsoid = isce3.core.WGS84_ELLIPSOID

    # Get orbit data from RSLC product if no external orbit was provided.
    if external_orbit is None:
        orbit = rslc.getOrbit()
    else:
        orbit = external_orbit

    # Estimate the absolute calibration error (the ratio of the measured RCS to the
    # predicted RCS) for a single corner reflector.
    def estimate_abscal_error(
        cr: isce3.cal.TriangularTrihedralCornerReflector,
    ) -> float:
        predicted_rcs = isce3.cal.predict_triangular_trihedral_cr_rcs(
            cr=cr,
            orbit=orbit,
            doppler=native_doppler,
            wavelength=radar_grid.wavelength,
            look_side=radar_grid.lookside,
        )

        measured_rcs = isce3.cal.measure_target_rcs(
            target_llh=cr.llh,
            img_data=img_data,
            radar_grid=radar_grid,
            orbit=orbit,
            doppler=image_grid_doppler,
            ellipsoid=ellipsoid,
            nchip=nchip,
            upsample_factor=upsample_factor,
            peak_find_domain=peak_find_domain,
            nfit=nfit,
            power_method=power_method,
            pthresh=pthresh,
        )

        return measured_rcs / predicted_rcs

    # Estimate the absolute radiometric calibration error of each corner reflector, and
    # format the results into an object that can be easily JSON-ified.
    results = []
    for cr in corner_reflectors:
        try:
            abscal_error = estimate_abscal_error(cr)
        except Exception:
            errmsg = traceback.format_exc()
            warnings.warn(
                f"an exception occurred while processing corner reflector {cr.id!r}:"
                f"\n\n{errmsg}",
                RuntimeWarning,
            )
            continue

        # TODO: Placeholder for now. To be implemented in R4.
        elevation_angle = np.nan

        # TODO: Update 'timestamp' to be the actual observation time of the corner
        # reflector from geo2rdr().
        cr_info = {
            "id": cr.id,
            "absolute_calibration_factor": abscal_error,
            "elevation_angle": elevation_angle,
            "timestamp": rslc.identification.zdStartTime,
            "frequency": freq,
            "polarization": pol,
        }

        # Add NISAR-specific metadata, if available.
        if isinstance(cr, nisar.cal.CornerReflector):
            cr_info.update(
                {"survey_date": cr.survey_date, "velocity": list(cr.velocity)}
            )

        results.append(cr_info)

    return results


def parse_cmdline_args() -> dict[str, Any]:
    """
    Parse command line arguments.

    Returns
    -------
    kwargs : dict
        A dict of keyword arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the absolute radiometric calibration factor of an RSLC product"
            " with one or more corner reflectors (CRs)."
        ),
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--crs",
        required=True,
        type=str,
        dest="corner_reflector_csv",
        help=(
            "A CSV file containing corner reflector data, in the format defined by the"
            " --format flag."
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        dest="csv_format",
        choices=["nisar", "uavsar"],
        default="nisar",
        help=(
            "The corner reflector CSV file format. If 'nisar', the CSV file should be"
            " in the format described by the NISAR Corner Reflector Software Interface"
            " Specification (SIS) document, JPL D-107698. If 'uavsar', the CSV file is"
            " expected to be in the format used by the UAVSAR Rosamond Corner Reflector"
            " Array (https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl)."
        ),
    )
    parser.add_argument(
        "--rslc",
        required=True,
        type=str,
        dest="rslc_hdf5",
        help="A NISAR RSLC product filename.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        dest="output_json",
        default=None,
        help=(
            "An output JSON file to write the results to. The file's parent directory"
            " will be created if it does not exist. If the file exists, it will be"
            " overwritten. If no output file is specified, the results will be written"
            " to the standard output stream."
        ),
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=str,
        choices=["A", "B"],
        default=None,
        help=(
            "The frequency sub-band of the data. Defaults to the science band in the"
            " RSLC product ('A' if available, otherwise 'B')."
        ),
    )
    parser.add_argument(
        "-p",
        "--pol",
        type=str,
        choices=["HH", "VV", "HV", "VH"],
        default=None,
        help=(
            "The transmit and receive polarization of the data. Defaults to the first"
            " co-polarization channel found in the specified band ('HH' if available,"
            " otherwise 'VV')."
        ),
    )
    parser.add_argument(
        "--orbit",
        type=str,
        dest="external_orbit_xml",
        default=None,
        help=(
            "Filename of an external orbit XML file to use instead of the orbit data"
            " contained within the RSLC product."
        ),
    )
    parser.add_argument(
        "--nchip",
        type=int,
        default=64,
        help=(
            "The width, in pixels, of the square block of image data to extract"
            " centered around the target position for oversampling and peak finding."
            " Must be >= 1."
        ),
    )
    parser.add_argument(
        "--upsample",
        type=int,
        dest="upsample_factor",
        default=32,
        help="The upsampling ratio. Must be >= 1.",
    )
    parser.add_argument(
        "--peak-find-domain",
        type=str,
        choices=["time", "freq"],
        default="time",
        help=(
            "Option controlling how the target peak position is estimated (time domain"
            " or frequency domain)."
        ),
    )
    parser.add_argument(
        "--nfit",
        type=int,
        default=5,
        help=(
            "The width, in *oversampled* pixels, of the square sub-block of image data"
            " to extract centered around the target position for fitting a quadratic"
            " polynomial to the peak. Note that this is the size in pixels *after"
            " upsampling*. Must be >= 3."
        ),
    )
    parser.add_argument(
        "--power-method",
        type=str,
        choices=["box", "integrated"],
        default="box",
        help=(
            "The method for estimating the target signal power (rectangular box method"
            " or integrated power method)."
        ),
    )
    parser.add_argument(
        "--pthresh",
        type=float,
        default=3.0,
        help=(
            "The minimum power threshold, measured in dB below the peak power, for"
            " estimating the target signal power using the integrated power method."
            " Only used if --power-method=integrated."
        ),
    )

    return vars(parser.parse_args())


def main(
    corner_reflector_csv: os.PathLike,
    csv_format: str,
    rslc_hdf5: os.PathLike,
    output_json: Optional[str],
    freq: Optional[str],
    pol: Optional[str],
    external_orbit_xml: Optional[os.PathLike],
    nchip: int,
    upsample_factor: int,
    peak_find_domain: str,
    nfit: int,
    power_method: str,
    pthresh: float,
) -> None:
    """Main entry point. See `parse_cmdline_args()` for parameter descriptions."""
    # Read input RSLC product.
    rslc_hdf5 = os.fspath(rslc_hdf5)
    rslc = nisar.products.readers.SLC(hdf5file=rslc_hdf5)

    # Get corner reflector data.
    if csv_format == "nisar":
        # Parse the input corner reflector CSV file. Filter out unusable corner
        # reflector data based on survey date and validity flags.
        corner_reflectors = nisar.cal.parse_and_filter_corner_reflector_csv(
            corner_reflector_csv,
            observation_date=rslc.identification.zdStartTime,
            validity_flags=nisar.cal.CRValidity.RAD_POL,
        )
    elif csv_format == "uavsar":
        # Parse the input corner reflector CSV file. No filtering is performed.
        corner_reflectors = isce3.cal.parse_triangular_trihedral_cr_csv(
            corner_reflector_csv
        )
    else:
        raise ValueError(f"unexpected csv format: {csv_format!r}")

    # Filter CRs by heading.
    approx_cr_heading = nisar.cal.est_cr_az_mid_swath_from_slc(rslc)
    corner_reflectors = nisar.cal.filter_crs_per_az_heading(
        corner_reflectors, az_heading=approx_cr_heading
    )

    # Parse external orbit XML file, if applicable.
    if external_orbit_xml is None:
        external_orbit = None
    else:
        external_orbit = nisar.products.readers.orbit.load_orbit_from_xml(
            external_orbit_xml
        )

    # Estimate abscal factor.
    results = estimate_abscal_factor(
        corner_reflectors=corner_reflectors,
        rslc=rslc,
        freq=freq,
        pol=pol,
        external_orbit=external_orbit,
        nchip=nchip,
        upsample_factor=upsample_factor,
        peak_find_domain=peak_find_domain,
        nfit=nfit,
        power_method=power_method,
        pthresh=pthresh,
    )

    if output_json is None:
        # Print results to console in JSON format.
        print(json.dumps(results, indent=2, cls=DateTimeEncoder))
    else:
        output_json = Path(output_json)

        # Recursively create output file's parent directories if they didn't already
        # exist.
        output_json.parent.mkdir(parents=True, exist_ok=True)

        # Write results to file in JSON format. Overwrite file if it exists.
        with output_json.open("w") as f:
            json.dump(results, f, indent=2, cls=DateTimeEncoder)


if __name__ == "__main__":
    main(**parse_cmdline_args())
