#!/usr/bin/env python3
"""
Analyze point targets in an RSLC HDF5 file
"""
from __future__ import annotations

import os
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

import nisar
from nisar.cal import CRValidity

import numpy as np
import argparse
import isce3
from nisar.products.readers import SLC
from isce3.core.types import ComplexFloat16Decoder
from isce3.cal import point_target_info as pti
import warnings
import json
from nisar.products.readers.GenericProduct import get_hdf5_file_product_type
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

desc = __doc__

def cmd_line_parse():
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-i", 
        "--input", 
        dest="input_file", 
        type=str, 
        required=True, 
        help="Input RSLC directory+filename"
    )
    parser.add_argument(
        "-f",
        "--freq",
        dest="freq_group",
        type=str,
        required=True,
        choices=["A", "B"],
        help="Frequency group in RSLC H5 file: A or B",
    )
    parser.add_argument(
        "-p",
        "--polarization",
        dest="pol",
        type=str,
        required=True,
        choices=["HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV"],
        help="Tx and Rx polarizations in RSLC H5 file: HH, HV, VH, VV",
    )

    # Add a required XOR group for the `corner_reflector_csv` and `geo_llh` parameters
    # so that one or the other (but not both) must be specified.
    cr_group = parser.add_mutually_exclusive_group(required=True)
    cr_group.add_argument(
        "--csv",
        type=Path,
        dest="corner_reflector_csv",
        help=(
            "A CSV file containing corner reflector data, in the format defined by the"
            " --format flag. Required if -c (--LLH) is not specified."
        ),
    )
    cr_group.add_argument(
        "-c",
        "--LLH",
        nargs=3,
        dest="geo_llh",
        type=float,
        help=(
            "Geodetic coordinates (longitude in degrees, latitude in degrees,"
            " height above ellipsoid in meters). Required if --csv is not specified."
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
            " Array (https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl). This flag is"
            " ignored if --csv was not specified."
        ),
    )
    parser.add_argument(
        "--fs-bw-ratio",
        type=float,
        default=1.2,
        required=False,
        help="Sampling Frequency to bandwidth ratio. Only used when --predict-null requested.",
    )
    parser.add_argument(
        "--num-sidelobes",
        type=int,
        default=10,
        required=False,
        help="number of sidelobes to be included for ISLR and PSLR computation"
    )
    parser.add_argument(
        "--plots",
        action='store_true',
        help="Generate interactive plots"
    )
    parser.add_argument(
        "--cuts",
        action="store_true",
        help="Add range/azimuth slices to output JSON."
    )
    parser.add_argument(
        "--nov", 
        type=int, 
        default=32, 
        help="Point target samples upsampling factor"
    )
    parser.add_argument(
        "--chipsize", 
        type=int, 
        default=64, 
        help="Point target chip size"
    )
    parser.add_argument(
        "-n",
        "--predict-null",
        action="store_true",
        help="If true, locate mainlobe nulls based on Fs/BW ratio instead of search",
    )
    parser.add_argument(
        "--window_type", 
        type=str, 
        default='rect', 
        help="Point target impulse reponse window tapering type: rect, kaiser, cosine. Only used when --predict-null requested."
    )
    parser.add_argument(
        "--window_parameter", 
        type=float,
        default=0.0,
        help="Point target impulse reponse window tapering parameter."
    )
    parser.add_argument(
        "--shift-domain",
        choices=("time", "frequency"),
        default="time",
        help="Estimate shift in time domain or frequency domain."
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=str,
        default=None,
        help="Output directory+filename of JSON file to which performance metrics are written. "
             "If None, then print performance metrics to screen.",
    )

    return parser.parse_args()

def get_radar_grid_coords(llh_deg, slc, freq_group):
    """Perform geo2rdr conversion of longitude, latitude, and
    height-above-ellipsoid (LLH) coordinates to radar geometry, and return the
    resulting coordinates with respect to the input product's radar grid.

    Parameters
    ------------
    llh_deg: array of 3 floats
        Corner reflector geodetic coordinates in lon (deg), lat (deg), and height (m)
        array size = 3
    slc: nisar.products.readers.SLC
        NISAR RSLC HDF5 product data 
    freq_group: str
       RSLC data file frequency selection 'A' or 'B'

    Returns
    --------
    pixel: float
        Point target slant range bin location
    line: float
        Point target azimuth time bin location
    """

    llh = np.array([np.deg2rad(llh_deg[0]), np.deg2rad(llh_deg[1]), llh_deg[2]])

    # Assume we want the WGS84 ellipsoid (a common assumption in isce3) 
    # and the radar grid is zero Doppler (always the case for NISAR products).
    ellipsoid = isce3.core.Ellipsoid()
    doppler = isce3.core.LUT2d()

    radargrid = slc.getRadarGrid(freq_group)
    orbit = slc.getOrbit()

    if radargrid.ref_epoch != orbit.reference_epoch:
        raise ValueError('Reference epoch of radar grid and orbit are different!')
        
    aztime, slant_range = isce3.geometry.geo2rdr(
        llh,
        ellipsoid,
        orbit,
        doppler,
        radargrid.wavelength,
        radargrid.lookside,
    )

    line = (aztime - radargrid.sensing_start) * radargrid.prf
    pixel = (slant_range - radargrid.starting_range) / radargrid.range_pixel_spacing

    return pixel, line


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for point_target_analysis output."""

    def default(self, obj):
        # Convert `DateTime` objects to ISO-8601 strings.
        if isinstance(obj, isce3.core.DateTime):
            return obj.isoformat()

        # Convert `CRValidity` flags to integers.
        if isinstance(obj, CRValidity):
            return int(obj)

        # Convert NumPy floating-point scalars to floats and arrays to lists.
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


def to_json(
    obj: Any,
    output: str | os.PathLike | None,
    encoder: type[json.JSONEncoder] | None = None,
) -> None:
    """
    Serialize `obj` to a JSON-formatted file.

    Parameters
    ----------
    obj : object
        The object to encode in JSON format.
    output : path-like or None
        File path of the JSON file to write the results to. The file's parent directory
        will be created if it does not exist. If the file exists, it will be
        overwritten. If None, the JSON string will be streamed to standard output
        instead.
    encoder : type or None, optional
        An custom `json.JSONEncoder` subclass to use for serialization. The default of
        None uses the default encoder.
    """
    if output is None:
        # Print results to console in JSON format.
        print(json.dumps(obj, indent=2, cls=encoder))
    else:
        output = Path(output)

        # Recursively create output file's parent directories if they didn't already
        # exist.
        output.parent.mkdir(parents=True, exist_ok=True)

        # Write results to file in JSON format. Overwrite file if it exists.
        with output.open("w") as f:
            json.dump(obj, f, indent=2, cls=encoder)


def slc_pt_performance(
    slc_input,
    freq_group,
    polarization,
    cr_llh,
    fs_bw_ratio=1.2,
    num_sidelobes=10,
    predict_null=True,
    nov=32,
    chipsize=64,
    plots=False,
    cuts=False,
    window_type='rect',
    window_parameter=0,
    shift_domain='time',
    pta_output=None
):
    """This function runs point target performance test on Level-1 RSLC.

    Parameters:
    ------------
    slc_input: str
        NISAR RSLC HDF5 product directory+filename
    freq_group: str
        RSLC data file frequency selection 'A' or 'B'
    polarization: str
        RSLC data file Tx/Rx polarization selection in 'HH', 'HV', 'VH', and 'VV'
    cr_llh: array of 3 floats
        Corner reflector geodetic coordinates in lon (deg), lat (deg), and height (m)
    fs_bw_ratio: float
        Sampling Frequency to bandwidth ratio
    num_sidelobes: int
        total number of sidelobes to be included in ISLR computation
    predict_null: bool
        optional, if predict_null is True, mainlobe null locations are computed based 
        on Fs/bandwidth ratio and mainlobe broadening factor for ISLR calculations.
        i.e, mainlobe null is located at Fs/B samples * broadening factor from the peak of mainlobe
        Otherwise, mainlobe null locations are computed based on null search algorithm.
        PSLR Exception: mainlobe does not include first sidelobes, search is always
        conducted to find the locations of first null regardless of predict_null parameter.
    nov: int
        Point target samples upsampling factor   
    chipsize: int
        Width in pixels of the square block, centered on the point target,
        used to estimate point target properties
    plots: bool
        Generate point target metrics plots
    cuts: bool
        Store range/azimuth cuts data to output JSON file
    window_type: str
        optional, user provided window types used for tapering
        
        'rect': 
                Rectangular window is applied
        'cosine': 
                Raised-Cosine window
        'kaiser': 
                Kaiser Window
    window_parameter: float
        optional window parameter. For a Kaiser window, this is the beta
        parameter. For a raised cosine window, it is the pedestal height.
        It is ignored if `window_type = 'rect'`.
    shift_domain: {time, frequency}
        If 'time' then estimate peak location using max of oversampled data.
        If 'frequency' then estimate a phase ramp in the frequency domain.
        Default is 'time' but 'frequency' is useful when target is well
        focused, has high SNR, and is the only target in the neighborhood
        (often the case in point target simulations).
    pta_output: str
        point target metrics output JSON file (directory+filename)

    Returns:
    --------
    performance_dict: dict
        Corner reflector performance output dictionary: -3dB resolution (in samples), 
        PSLR (dB), ISLR (dB), slant range offset (in samples), and azimuth offset
        (in samples).
    """
  
    # Raise an exception if input is a GSLC HDF5 file
    product_type = get_hdf5_file_product_type(slc_input)
    if product_type == "GSLC":
        raise NotImplementedError("support for GSLC products is not yet implemented") 

    # Open RSLC data
    slc = SLC(hdf5file=slc_input)
    slc_data = ComplexFloat16Decoder(slc.getSlcDataset(freq_group, polarization))

    #Convert input LLH (lat, lon, height) coordinates into (slant range, azimuth)
    slc_pixel, slc_line = get_radar_grid_coords(cr_llh, slc, freq_group)

    #compute point target performance metrics
    performance_dict = pti.analyze_point_target(
        slc_data,
        slc_line,
        slc_pixel,
        nov,
        plots,
        cuts,
        chipsize,
        fs_bw_ratio,
        num_sidelobes,
        predict_null,
        window_type,
        window_parameter,
        shift_domain=shift_domain,
    )
   
    if plots:
        performance_dict = performance_dict[0] 
    
    # Write dictionary content to a json file if output is requested by user
    to_json(performance_dict, pta_output, encoder=CustomJSONEncoder)

    return performance_dict


CornerReflectorIterable = Union[
    Iterable[isce3.cal.TriangularTrihedralCornerReflector],
    Iterable[nisar.cal.CornerReflector],
]


def analyze_corner_reflectors(
    corner_reflectors: CornerReflectorIterable,
    rslc: nisar.products.readers.SLC,
    freq: str | None = None,
    pol: str | None = None,
    *,
    nchip: int = 64,
    upsample_factor: int = 32,
    peak_find_domain: str = "time",
    num_sidelobes: int = 10,
    predict_null: bool = True,
    fs_bw_ratio: float = 1.2,
    window_type: str = "rect",
    window_parameter: float = 0.0,
    cuts: bool = False,
) -> list[dict[str, Any]]:
    r"""
    Analyze corner reflector (CR) characteristics using RSLC data.

    Characterize the geolocation accuracy and impulse response function (IRF) of zero or
    more corner reflectors.

    If the processing fails for any input CR, the error message and traceback info will
    be logged and the CR will be omitted from the results.

    Parameters
    ----------
    corner_reflectors : iterable
        Iterable of corner reflectors in the scene. The elements may be instances of
        `isce3.cal.TriangularTrihedralCornerReflector` or `nisar.cal.CornerReflector`.
        In the latter case, additional information about the survey date, validity, and
        plate motion velocity of each corner reflector is populated in the output.
    rslc : nisar.products.readers.SLC
        The input RSLC product.
    freq : {'A', 'B'} or None, optional
        The frequency sub-band of the data. Defaults to the science band in the RSLC
        product ('A' if available, otherwise 'B').
    pol : {'HH', 'HV', 'VH', 'VV', 'LH', 'LV', 'RH', 'RV'} or None, optional
        The transmit and receive polarization of the data. Defaults to the first
        co-polarization or compact polarization channel found in the specified band from
        the list ['HH', 'VV', 'LH', 'LV', 'RH', 'RV'].
    nchip : int, optional
        The width, in pixels, of the square block of image data centered around the
        target position to extract for oversampling and peak finding. Must be >= 1.
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
          domain. This mode is useful when the target is well-focused, has high SNR, and
          is the only target in the neighborhood (often the case in point target
          simulations).
    num_sidelobes : int, optional
        The number of sidelobes, including the main lobe, to use for computing the
        integrated sidelobe ratio (ISLR). Must be > 1. Defaults to 10.
    predict_null : bool, optional
        Controls how the main lobe null locations are determined for ISLR computation.
        If `predict_null` is true, the null locations are determined analytically by
        assuming that the corner reflector has the impulse response of a point target
        with known sampling-rate-to-bandwidth ratio (given by `fs_bw_ratio`) and range &
        azimuth spectral windows (given by `window_type` & `window_parameter`). In this
        case, the first sidelobe will be considered to be part of the main lobe.
        Alternatively, if `predict_null` is false, the apparent null locations will be
        estimated from the RSLC image data by searching for nulls in range & azimuth
        cuts centered on the target location. In this case, the main lobe does *not*
        include the first sidelobe. `predict_null` has no effect on peak-to-sidelobe
        ratio (PSLR) computation -- for PSLR analysis, the null locations are always
        determined by searching for nulls in the RSLC data. Defaults to True.
    fs_bw_ratio : float, optional
        The ratio of sampling rate to bandwidth in the RSLC image data. Must be the same
        for both range & azimuth. It is ignored if `predict_null` was false. Defaults to
        1.2 (the nominal oversampling ratio of NISAR RSLC data).
    window_type : {'rect', 'cosine', 'kaiser'}, optional
        The window type used in RSLC formation. Used to predict the locations of main
        lobe nulls during ISLR processing if `predict_null` was true. It is ignored if
        `predict_null` was false. The same window type is assumed to have been used for
        both range & azimuth focusing.

        'rect':
            The default. Assumes that the RSLC image was formed using a
            rectangular-shaped window (i.e. no spectral weighting was applied).
        'cosine':
            Assumes that the RSLC image was formed using a raised-cosine window with
            pedestal height defined by `window_parameter`.
        'kaiser':
            Assumes that the RSLC image was formed using a Kaiser window with beta
            parameter defined by `window_parameter`.
    window_parameter : float, optional
        The window shape parameter used in RSLC formation. The meaning of this parameter
        depends on the `window_type`. For a raised-cosine window, it is the pedestal
        height of the window. For a Kaiser window, it is the beta parameter. It is
        ignored if `window_type` was 'rect' or if `predict_null` was false. The same
        shape parameter is assumed to have been used for both range & azimuth focusing.
        Defaults to 0.
    cuts : bool, optional
        Whether to include range & azimuth cuts through the peak in the results.
        Defaults to False.

    Returns
    -------
    results : list of dict
        A list of (nested) dicts containing one entry per corner reflector found in the
        scene. The dict of results for each corner reflector consists of the following
        keys:

        'id':
          The unique identifier of the corner reflector.

        'frequency':
          The frequency sub-band of the data.

        'polarization':
          The transmit and receive polarization of the data.

        'magnitude':
          The peak magnitude of the impulse response.

        'phase':
          The phase at the peak location, in radians.

        'azimuth':
          A dict containing info about the azimuth impulse response function, consisting
          of the following keys:

            'index':
              The real-valued azimuth index, in samples, of the estimated peak location
              of the IRF within the RSLC image grid.

            'offset':
              The error in the predicted target location in the azimuth direction, in
              samples. Equal to the signed difference between the measured location of
              the IRF peak in the RSLC data and the predicted location of
              the peak based on the surveyed corner reflector location.

            'phase ramp':
              The estimated azimuth phase slope at the target location, in radians per
              sample.

            'resolution':
              The measured 3dB width of the azimuth IRF, in samples.

            'ISLR':
              The integrated sidelobe ratio of the azimuth IRF, in decibels (dB). A
              measure of the ratio of energy in the sidelobes to the energy in the main
              lobe. If `predict_null` was true, the first sidelobe will be considered
              part of the main lobe and the ISLR will instead measure the ratio of
              energy in the remaining sidelobes to the energy in the main lobe + first
              sidelobe.

            'PSLR':
              The peak-to-sidelobe ratio of the azimuth IRF, in decibels (dB). A measure
              of the ratio of peak sidelobe power to the peak main lobe power.

        'range':
          A dict containing info about the range impulse response function, with the
          same structure as the 'azimuth' dict.

        If `cuts` was true, the 'azimuth' and 'range' dicts each contain additional
        keys:

        'magnitude cut':
          The magnitude of the (upsampled) impulse response function in azimuth/range.

        'phase cut':
          The phase of the (upsampled) impulse response function in azimuth/range.

        'cut':
          The azimuth/range sample indices of the magnitude and phase cut values.

        If the input corner reflectors were instances of `nisar.cal.CornerReflector`,
        the following additional keys are also populated in the top-level dict for each
        corner reflector:

        'survey_date':
          The date (and time) when the corner reflector was surveyed most recently prior
          to the radar observation.

        'validity':
          The integer validity code of the corner reflector. Refer to the NISAR Corner
          Reflector Software Interface Specification (SIS) document\ [1]_ for details.

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
    .. [1] B. Hawkins, "Corner Reflector Software Interface Specification," JPL
       D-107698 (2023).
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
        for p in ["HH", "VV", "LH", "LV", "RH", "RV"]:
            if p in available_pols:
                pol = p
                break
        else:
            raise ValueError(
                f"no co-pols or compact pols found in freq {freq!r} of RSLC product."
                f" Available polarizations are {set(available_pols)}"
            )
    else:
        possible_pols = {"HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV"}
        if pol not in possible_pols:
            raise ValueError(f"pol must be in {possible_pols} (or None), got {pol!r}")
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

    # Returns point target info dict for a single target, given its location in geodetic
    # coordinates.
    def get_point_target_info(target_llh: isce3.core.LLH) -> dict[str, Any]:
        # Convert lon & lat to degrees.
        lon, lat, height = target_llh.to_vec3()
        llh_deg = np.asarray([np.rad2deg(lon), np.rad2deg(lat), height])

        # Get pixel-space coordinates of the target within the image grid.
        col, row = get_radar_grid_coords(llh_deg, rslc, freq)

        # Get point target info.
        return pti.analyze_point_target(
            img_data,
            row,
            col,
            nov=upsample_factor,
            plot=False,
            cuts=cuts,
            chipsize=nchip,
            fs_bw_ratio=fs_bw_ratio,
            num_sidelobes=num_sidelobes,
            predict_null=predict_null,
            window_type=window_type,
            window_parameter=window_parameter,
            shift_domain=peak_find_domain,
        )

    orbit = rslc.getOrbit()
    attitude = rslc.getAttitude()
    radar_grid = rslc.getRadarGrid(freq)

    results = []
    for cr in corner_reflectors:
        try:
            cr_info = get_point_target_info(cr.llh)
        except Exception:
            errmsg = traceback.format_exc()
            warnings.warn(
                f"an exception occurred while processing corner reflector {cr.id!r}:"
                f"\n\n{errmsg}",
                RuntimeWarning,
            )
            continue

        # Make sure we're not overwriting any info in the dict.
        assert "id" not in cr_info
        assert "frequency" not in cr_info
        assert "polarization" not in cr_info
        assert "elevation_angle" not in cr_info
        assert "survey_date" not in cr_info
        assert "validity" not in cr_info
        assert "velocity" not in cr_info

        # Get the target's zero-Doppler elevation angle.
        _, elevation_angle = isce3.cal.get_target_observation_time_and_elevation(
            target_llh=cr.llh,
            orbit=orbit,
            attitude=attitude,
            wavelength=radar_grid.wavelength,
            look_side=radar_grid.lookside,
        )

        # Add some additional metadata.
        cr_info.update(
            {
                "id": cr.id,
                "frequency": freq,
                "polarization": pol,
                "elevation_angle": elevation_angle,
            }
        )

        # Add NISAR-specific corner reflector metadata, if available.
        if isinstance(cr, nisar.cal.CornerReflector):
            cr_info.update(
                {
                    "survey_date": cr.survey_date,
                    "validity": cr.validity,
                    "velocity": cr.velocity,
                }
            )

        results.append(cr_info)

    return results


def process_corner_reflector_csv(
    corner_reflector_csv: str | os.PathLike,
    csv_format: str,
    rslc_hdf5: str | os.PathLike,
    output_json: str | os.PathLike | None,
    freq: str | None,
    pol: str | None,
    nchip: int,
    upsample_factor: int,
    peak_find_domain: str,
    num_sidelobes: int,
    predict_null: bool,
    fs_bw_ratio: float,
    window_type: str,
    window_parameter: float,
    cuts: bool,
) -> None:
    """
    Run point target analysis on corner reflectors from a CSV file.

    Parameters
    ----------
    corner_reflector_csv : path-like
        A CSV file containing corner reflector data.
    csv_format : {'nisar', 'uavsar'}
        The corner reflector CSV file format. If 'nisar', the CSV file should be in the
        format described by the NISAR Corner Reflector Software Interface Specification
        (SIS) document, JPL D-107698. If 'uavsar', the CSV file is expected to be in the
        format used by the UAVSAR Rosamond Corner Reflector Array
        (https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl).
    rslc_hdf5 : path-like
        A NISAR RSLC product file name.
    output_json : path-like or None
        An output JSON file to write the results to. The file's parent directory will be
        created if it does not exist. If the file exists, it will be overwritten. If no
        output file is specified, the results will be written to the standard output
        stream.
    freq : {'A', 'B'} or None
        The frequency sub-band of the data. If None, defaults to the science band in the
        RSLC product ('A' if available, otherwise 'B').
    pol : {'HH', 'HV', 'VH', 'VV', 'LH', 'LV', 'RH', 'RV'} or None
        The transmit and receive polarization of the data. If None, defaults to the
        first co-polarization or compact polarization channel found in the specified
        band from the list ['HH', 'VV', 'LH', 'LV', 'RH', 'RV'].
    nchip : int
        The width, in pixels, of the square block of image data centered around the
        target position to extract for oversampling and peak finding. Must be >= 1.
    upsample_factor : int
        The upsampling ratio. Must be >= 1.
    peak_find_domain : {'time', 'freq'}
        Option controlling how the target peak position is estimated.

        'time':
          The peak location is found in the time domain by detecting the maximum value
          within a square block of image data around the expected target location. The
          signal data is upsampled to improve precision.

        'freq':
          The peak location is found by estimating the phase ramp in the frequency
          domain. This mode is useful when the target is well-focused, has high SNR, and
          is the only target in the neighborhood (often the case in point target
          simulations).
    num_sidelobes : int
        The number of sidelobes, including the main lobe, to use for computing the
        integrated sidelobe ratio (ISLR). Must be > 1.
    predict_null : bool
        Controls how the main lobe null locations are determined for ISLR computation.
        If `predict_null` is true, the null locations are determined analytically by
        assuming that the corner reflector has the impulse response of a point target
        with known sampling-rate-to-bandwidth ratio (given by `fs_bw_ratio`) and range &
        azimuth spectral windows (given by `window_type` & `window_parameter`). In this
        case, the first sidelobe will be considered to be part of the main lobe.
        Alternatively, if `predict_null` is false, the apparent null locations will be
        estimated from the RSLC image data by searching for nulls in range & azimuth
        cuts centered on the target location. In this case, the main lobe does *not*
        include the first sidelobe. `predict_null` has no effect on peak-to-sidelobe
        ratio (PSLR) computation -- for PSLR analysis, the null locations are always
        determined by searching for nulls in the RSLC data.
    fs_bw_ratio : float
        The ratio of sampling rate to bandwidth in the RSLC image data. Must be the same
        for both range & azimuth. It is ignored if `predict_null` was false.
    window_type : {'rect', 'cosine', 'kaiser'}
        The window type used in RSLC formation. Used to predict the locations of main
        lobe nulls during ISLR processing if `predict_null` was true. It is ignored if
        `predict_null` was false. The same window type is assumed to have been used for
        both range & azimuth focusing.

        'rect':
            Assumes that the RSLC image was formed using a rectangular-shaped window
            (i.e. no spectral weighting was applied).
        'cosine':
            Assumes that the RSLC image was formed using a raised-cosine window with
            pedestal height defined by `window_parameter`.
        'kaiser':
            Assumes that the RSLC image was formed using a Kaiser window with beta
            parameter defined by `window_parameter`.
    window_parameter : float
        The window shape parameter used in RSLC formation. The meaning of this parameter
        depends on the `window_type`. For a raised-cosine window, it is the pedestal
        height of the window. For a Kaiser window, it is the beta parameter. It is
        ignored if `window_type` was 'rect' or if `predict_null` was false. The same
        shape parameter is assumed to have been used for both range & azimuth focusing.
    cuts : bool
        Whether to include range & azimuth cuts through the peak in the results.
    """
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
            validity_flags=(CRValidity.IPR | CRValidity.GEOM),
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

    # Analyze point targets.
    results = analyze_corner_reflectors(
        corner_reflectors=corner_reflectors,
        rslc=rslc,
        freq=freq,
        pol=pol,
        nchip=nchip,
        upsample_factor=upsample_factor,
        peak_find_domain=peak_find_domain,
        num_sidelobes=num_sidelobes,
        predict_null=predict_null,
        fs_bw_ratio=fs_bw_ratio,
        window_type=window_type,
        window_parameter=window_parameter,
        cuts=cuts,
    )

    # Serialize results to JSON.
    to_json(results, output_json, encoder=CustomJSONEncoder)


if __name__ == "__main__":
    inputs = cmd_line_parse()
    slc_input = inputs.input_file
    pta_output = inputs.output_file
    freq_group = inputs.freq_group
    pol = inputs.pol
    corner_reflector_csv = inputs.corner_reflector_csv
    cr_llh = inputs.geo_llh
    csv_format = inputs.csv_format
    fs_bw_ratio = inputs.fs_bw_ratio
    predict_null = inputs.predict_null
    num_sidelobes = inputs.num_sidelobes
    nov = inputs.nov
    chipsize = inputs.chipsize
    plots = inputs.plots
    cuts = inputs.cuts
    window_type = inputs.window_type
    window_parameter = inputs.window_parameter

    if (corner_reflector_csv is not None) and (cr_llh is None):
        # The user provided a corner reflector CSV file.
        process_corner_reflector_csv(
            corner_reflector_csv=corner_reflector_csv,
            csv_format=csv_format,
            rslc_hdf5=slc_input,
            output_json=pta_output,
            freq=freq_group,
            pol=pol,
            nchip=chipsize,
            upsample_factor=nov,
            peak_find_domain=inputs.shift_domain,
            num_sidelobes=num_sidelobes,
            predict_null=predict_null,
            fs_bw_ratio=fs_bw_ratio,
            window_type=window_type,
            window_parameter=window_parameter,
            cuts=cuts,
        )
    elif (cr_llh is not None) and (corner_reflector_csv is None):
        # The user provided LLH coordinates for a single corner reflector.
        performance_dict = slc_pt_performance(
            slc_input,
            freq_group,
            pol,
            cr_llh,
            fs_bw_ratio,
            num_sidelobes,
            predict_null,
            nov,
            chipsize,
            plots,
            cuts,
            window_type,
            window_parameter,
            shift_domain = inputs.shift_domain,
            pta_output = pta_output,
        )

        if plots and plt is not None:
            plt.show()
    else:
        # Should be unreachable.
        assert False, "invalid arguments to point_target_analysis"
