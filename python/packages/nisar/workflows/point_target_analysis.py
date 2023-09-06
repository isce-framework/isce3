"""
Analyze point targets in an RSLC HDF5 file
"""
import numpy as np
import argparse
import isce3
from nisar.products.readers import SLC
from isce3.core.types import ComplexFloat16Decoder
from isce3.cal import point_target_info as pti
import warnings
import json
from nisar.products.readers.GenericProduct import get_hdf5_file_product_type
import matplotlib.pyplot as plt

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
    parser.add_argument(
        "-c",
        "--LLH",
        nargs=3,
        dest="geo_llh",
        type=float,
        required=True,
        help=(
            "Geodetic coordinates (longitude in degrees, latitude in degrees,"
            " height above ellipsoid in meters)"
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
        type=int, 
        default=0, 
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
    if pta_output != None:
        pti.tofloatvals(performance_dict)
        with open(pta_output, 'w') as f:
            json.dump(performance_dict, f, ensure_ascii=False, indent=2)
    # Print dictionary content to screen
    else:
        pti.tofloatvals(performance_dict)
        print(json.dumps(performance_dict, ensure_ascii=False, indent=2))

    return performance_dict

if __name__ == "__main__":
    inputs = cmd_line_parse()
    slc_input = inputs.input_file
    pta_output = inputs.output_file
    freq_group = inputs.freq_group
    pol = inputs.pol
    cr_llh = inputs.geo_llh
    fs_bw_ratio = inputs.fs_bw_ratio
    predict_null = inputs.predict_null
    num_sidelobes = inputs.num_sidelobes
    nov = inputs.nov
    chipsize = inputs.chipsize
    plots = inputs.plots
    cuts = inputs.cuts
    window_type = inputs.window_type
    window_parameter = inputs.window_parameter


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

    if plots:
        plt.show()
