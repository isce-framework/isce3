"""
Estimate noise power of BCAL/LCAL lines in L0B raw data
"""
import numpy as np
from numpy import linalg as la
import argparse
from nisar.products.readers.Raw import Raw
from nisar.antenna.antenna_pattern import CalPath
import json
import warnings

desc = __doc__


def cmd_line_parse():
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=str,
        required=True,
        help="Input file path",
    )
    parser.add_argument(
        "-f",
        "--freq",
        dest="freq_group",
        type=str,
        required=True,
        choices=["A", "B"],
        help="Frequency group in raw L0B H5 file: A or B",
    )
    parser.add_argument(
        "-p",
        "--polarizations",
        dest="pol",
        type=str,
        required=True,
        choices=["HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV"],
        help=(
            "Tx and Rx polarizations in raw L0B file:"
            " HH, HV, VH, VV, LH, LV, RH, or RV"
        ),
    )
    parser.add_argument(
        "--rng-start",
        dest="rng_start",
        type=int,
        default=None,
        nargs="+",
        required=False,
        help="Start range bins for beams",
    )
    parser.add_argument(
        "--rng-stop",
        dest="rng_stop",
        type=int,
        default=None,
        nargs="+",
        required=False,
        help="Stop range bins for beams",
    )
    parser.add_argument(
        "-c",
        "--cpi",
        type=int,
        default=2,
        required=False,
        help="""Coherent processing interval in number of range lines. 
              Only used when --algorithm='evd'.""",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        dest="algo",
        type=str,
        default="avg",
        required=False,
        help="""Noise Estimation algorithms: 'avg' - mean of sum of squares and 
                'evd' - Eigen Value Decompostion""",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        default=None,
        help="Output JSON file path that stores noise estimates",
    )

    return parser.parse_args()


class InsufficientPulses(UserWarning):
    """
    Raised if number of input range lines is less than number
    range lines in a CPI.
    """

    pass


def validate_rng_start_stop(rng_start, rng_stop, raw_data):
    """Validate input parameters and reformat.

    Parameters:
    -----------
    rng_start: int or list of int
        Start range bin of each beam
    rng_stop: int or list of int
        Stop range bin of each beam
    raw_data: 2D array of complex
        Raw data for noise power estimation

    Returns:
    --------
    rng_start: list of int
        Start range bin of each beam
    rng_stop: list of int
        Stop range bin of each beam
    num_samps_beam: list of int
        number samples in a beam
    num_beams: int
        number of beams after beamforming, maximum = 12
    """
    # handle the case where noise power estimate over a single beam is requested.
    if isinstance(rng_start, int) and isinstance(rng_stop, int):
        num_samps_beam = [rng_stop - rng_start + 1]
        rng_start = [rng_start]
        rng_stop = [rng_stop]
        num_beams = len(rng_start)
    # handle the case where rng_start and rng_stop are not provided by user.
    # In this case, the noise power is estimated over the entire rangeline.
    elif rng_start == None and rng_stop == None:
        num_samps_beam = [raw_data.shape[1]]
        rng_start = [0]
        rng_stop = num_samps_beam
        num_beams = len(rng_start)
    # handle the case where noise power estimates are requested for multiple beams.
    elif isinstance(rng_start, list) and isinstance(rng_stop, list):
        # Check if rng_start and rng_stop are of the same size
        if len(rng_start) != len(rng_stop):
            raise ValueError("size of rng_start and rng_stop should be the same")
        num_samps_beam = list(
            np.array(rng_stop).astype(int) - np.array(rng_start).astype(int) + 1
        )
        num_beams = len(rng_start)
    # Handle the case when rng_start and rng_stop are of different data types
    elif type(rng_start) != type(rng_stop):
        raise TypeError("Data type of rng_start and rng_stop are not identical")

    # Check if rng_start values are negative
    if not all(x >= 0 for x in rng_start):
        raise ValueError("rng_start value should be >0")

    # Check if rng_stop is greater than rng_start
    if any((np.array(rng_stop) - np.array(rng_start)) <= 0):
        raise ValueError("rng_stop value should be greater than rng_start")

    return (rng_start, rng_stop, num_samps_beam, num_beams)


def noise_est_avg(raw_data, rng_start=None, rng_stop=None):
    """Estimate noise power using sample square average.

    Parameters:
    -----------
    raw_data: 2d array of complex
        Raw data for noise power estimation
    rng_start: int or list of int
        start range bin of each beam
    rng_stop: int or list of int
        stop range bin of each beam

    Returns:
    --------
    noise_pwr_beam_db: float or array of float
        Mean noise power estimate of all samples or of each beam in db, scalar if num_beams = 1
    """

    # validate range start and range stop indices
    rng_start, rng_stop, num_samps_beam, num_beams = validate_rng_start_stop(
        rng_start, rng_stop, raw_data
    )

    num_lines = raw_data.shape[0]

    noise_pwr = np.zeros([num_lines, num_beams])
    for idx_line in range(num_lines):
        raw_line = raw_data[idx_line]
        for idx_beam in range(num_beams):
            raw_beam = raw_line[rng_start[idx_beam] : rng_stop[idx_beam]]
            pwr_beam = np.square(np.abs(raw_beam))
            pwr_beam_avg = np.mean(pwr_beam)

            noise_pwr[idx_line, idx_beam] = pwr_beam_avg

    if num_beams == 1:
        noise_pwr = np.squeeze(noise_pwr, axis=1)

    noise_pwr_beam_db = 10 * np.log10(np.mean(noise_pwr, axis=0))

    return noise_pwr_beam_db


def noise_est_evd(raw_data, cpi=2, rng_start=None, rng_stop=None):
    """Estimate Noise Power using Minimum Eigenvalues.
       1. Divide slow time range lines into coherent processing interval (CPI) blocks
       2. Perform Eigenvalue decomposition (EVD) on each CPI.
       3. Minimum eigenvalue of the CPI is the noise power estimate of all range lines in CPI.

    Parameters:
    -----------
    raw_data: 2D array of complex
        Raw data for noise power estimation
    cpi: int
        Number of range lines in a Coherent Processing Interval (CPI) which the algorithm constructs
        a sample covariance matrix, default = 2
    rng_start: int or list of int
        Start range bin of each beam
    rng_stop: int or list of int
        Stop range bin of each beam

    Returns:
    --------
    noise_pwr_beam_db: float or array of float
        Noise power estimate of all samples or of each beam in db, scalar if num_beams = 1
    """

    # validate beam start and stop indices
    rng_start, rng_stop, num_samps_beam, num_beams = validate_rng_start_stop(
        rng_start, rng_stop, raw_data
    )

    num_lines = raw_data.shape[0]
    num_cpi = int(num_lines / cpi)

    cpi_start = np.arange(0, num_lines, cpi)
    cpi_stop = cpi_start + cpi

    # If number of lines in the raw data is a small value
    if num_lines < 2:
        raise ValueError(
            "Total number range lines must be at least 2 for EVD algorithm"
        )
    elif num_lines < cpi:
        warnings.warn(
            "Number of input pulses is less than that of CPI. CPI is changed to number of input pulses.",
            InsufficientPulses,
        )
        num_cpi = 1
        cpi_start = [0]
        cpi_stop = [num_lines]
    elif num_lines > cpi:
        # Check to see if there is a partial block.
        # Shift cpi_start[-1] and cpi_stop[-1] form a full CPI block

        if cpi_start[-1] + cpi > num_lines:
            cpi_start[-1] = num_lines - cpi
            cpi_stop[-1] = cpi_start[-1] + cpi

    noise_pwr_ev_min_beam = np.zeros((num_cpi, num_beams))

    for idx_cpi in range(num_cpi):
        data_cpi = raw_data[cpi_start[idx_cpi] : cpi_stop[idx_cpi]]

        # Construct sample covariance matrix and perform Eigenvalue Decomposition
        for idx_beam in range(num_beams):
            beam_cpi = data_cpi[:, rng_start[idx_beam] : rng_stop[idx_beam]]
            cov_cpi_beam = (
                np.matmul(beam_cpi, np.conj(beam_cpi).transpose())
                / num_samps_beam[idx_beam]
            )
            ev_min = eigen_decomp(cov_cpi_beam)

            noise_pwr_ev_min_beam[idx_cpi, idx_beam] = np.abs(ev_min)

    if num_beams == 1:
        noise_pwr_ev_min_beam = np.squeeze(noise_pwr_ev_min_beam, axis=1)

    noise_pwr_beam_db = 10 * np.log10(np.mean(noise_pwr_ev_min_beam, axis=0))

    return noise_pwr_beam_db


def eigen_decomp(cov_matrix):
    """Compute Minimum Eigenvalue of sample covariance matrix

    Parameters:
    -----------
    cov_matrix: array of 2D complex
        Covariance Matrix of input raw data lines

    Returns:
    --------
    ev_min:  float
        Minimum eigenvalue as noise power estimate in linear value
    """

    eig_val, eig_vec = la.eig(cov_matrix)

    ev_min = np.amin(eig_val)

    return ev_min


def extract_cal_lines(data_input, freq_group, pol):
    """Extract BCAL and LCAL lines from a L0B raw data set

    Parameters:
    -----------
    data_input: str
        Raw L0B data file path
    freq_group: str
        L0B raw data file frequency selection 'A' or 'B'
    pol: str
        L0B raw data file Tx and Rx polarization selection 'HH', 'HV', 'VH', 'VV', 'LH', 'LV', 'RH', 'RV'

    Returns:
    --------
    raw_cal_lines: complex array of 2D
         BCAL and LCAL range line data
    """
    raw = Raw(hdf5file=data_input)
    raw_data = raw.getRawDataset(freq_group, pol)

    cal_path_mask = raw.getCalType(freq_group, tx=pol[0])
    cal_lines_idx = (cal_path_mask == CalPath.BYPASS) | (
        cal_path_mask == CalPath.LNA
    ).astype(int)

    return raw_data[cal_lines_idx.nonzero()[0], :]


if __name__ == "__main__":
    inputs = cmd_line_parse()

    data_input = inputs.input_path
    freq_group = inputs.freq_group
    cpi = inputs.cpi
    pol = inputs.pol
    rng_start = inputs.rng_start
    rng_stop = inputs.rng_stop
    algo = inputs.algo
    output_file = inputs.output_path

    # Get indices of rangelines that correspond to BCAL and LCAL measurements
    raw_cal_lines = extract_cal_lines(data_input, freq_group, pol)
    num_lines, num_samples = raw_cal_lines.shape

    # Check number of Cal range lines.
    if num_lines == 0:
        raise ValueError("There are no CAL line in this raw data set.")
    elif num_lines == 1 and algo != "avg":
        raise ValueError(
            "Only one Cal pulse found, which is too few for 'evd'"
            " method. Try 'avg' instead or supply more data."
        )

    # Run Sample Mean or EVD methods
    if algo == "evd":
        noise_pwr_beam_db = noise_est_evd(raw_cal_lines, cpi, rng_start, rng_stop)
    elif algo == "avg":
        noise_pwr_beam_db = noise_est_avg(raw_cal_lines, rng_start, rng_stop)
    else:
        raise ValueError("Unrecognized algorithm.")

    # Generate output JSON file
    noise_pwr_dictionary = {
        "Noise Power Estimate(s) dB": noise_pwr_beam_db.tolist(),
        "Estimation Method": algo.upper(),
    }
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(noise_pwr_dictionary, f, ensure_ascii=False, indent=2)

    print(json.dumps(noise_pwr_dictionary, ensure_ascii=False, indent=2))
