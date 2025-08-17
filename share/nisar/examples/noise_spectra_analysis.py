"""
Analyze noise-only receive power spectra. Collect spectral information with regards to
RFI in terms of power and frequency locations.
"""

import numpy as np
import argparse
import h5py
from scipy.fft import fftshift
from scipy.signal import welch
from numpy import linalg as la
from nisar.products.readers.Raw import Raw
from isce3.signal.compute_evd_cpi import slice_gen
import os


def cmd_line_parse():
    parser = argparse.ArgumentParser(description='Perform spectra analysis on input L0B file.')

    parser.add_argument(
        '-i',
        '--input', 
        dest='input_file', 
        type=str,
        required=True, 
        help='Input L0B file'
    )
    parser.add_argument(
        '-o',
        '--output', 
        dest='output_dir', 
        type=str,
        required=True, 
        help='Output data path. Output file name is input file name + SPECTRA.h5'
    )
    parser.add_argument(
        "-p",
        "--proc-interval",
        dest="num_pulses_proc",
        type=int,
        required=False,
        default="6",
        help="Number of pulses to be used to estimate signle pulse power. Default: 6",
    )
    parser.add_argument(
        "-n",
        "--nfft",
        dest="num_fft",
        type=int,
        default="2048",
        help="Number of samples per segment for Scipy Welch function." 
             "It is equal to number of FFT to be performed on each segment."
             "Default: 2048"
    )
    parser.add_argument(
        "-b",
        "--block-size",
        dest="num_pulses_blk",
        type=int,
        required=False,
        default="4096",
        help="Number of pulses per batch to be averaged in Azimuth-Time. Default: 5000",
    )
    parser.add_argument(
        "-m",
        "--margin",
        dest="thresh_margin",
        type=float,
        required=False,
        default="14",
        help="Additional gain to be added to noise power estimate for RFI thresholding." 
             "Default: 6"
    )

    return parser.parse_args() 


def select_pulses_for_estimation(
    raw_data: np.ndarray, 
    num_pulses_proc: int = 6
):
    """Select pulses for Eigenvalue Decomposition based on num_pulses_proc

    Parameters
    ----------
    raw_data: complex np.ndarray
        complex input raw data
    num_pulses_proc: int
        Number of pulses to be used to estimate (noise power) of a single pulse

    Returns
    -------
    raw_data_selcted: 2D complex array
        Selected receive pulses for sample covariance estimation and Eigenvalue
        Decomposition
    """

    num_pulses = raw_data.shape[0]

    if num_pulses_proc > num_pulses:
        raise ValueError(f"Requested num_pulses_proc={num_pulses_proc} exceeds number of pulses {num_pulses}")
    
    # Equally spaced indices (without repetition)
    indices = np.linspace(0, num_pulses - 1, num_pulses_proc, dtype=int)
    raw_data_selected = raw_data[indices, :]


    return raw_data_selected


def estimate_noise_pwr_pulse(
    raw_data: np.ndarray, 
    num_pulses_proc: int = 6
) -> float:
    """Estimate noise power per pulse in the time domain using minimum Eigenvalue

    Parameters
    ----------
    raw_data: complex np.ndarray
        complex input raw data
    num_pulses_proc: int
        Number of pulses to be used to estimate noise power of a single pulse
        Default: 6 

    Returns
    -------
    noise_pwr_pulse_est_db: float
        Estmated noise power per pulse using minimum Eigenvalue
    """

    data_estimate = select_pulses_for_estimation(raw_data, num_pulses_proc)
    sample_cov = data_estimate @ data_estimate.conj().T / data_estimate.shape[1]
    eig_vals, _ = la.eigh(sample_cov)

    noise_pwr_pulse_est_db = 10 * np.log10(eig_vals[0])

    return noise_pwr_pulse_est_db

def update_rfi_hit_stats(
    rfi_mask_blk, 
    rfi_freq_bin_hit_count, 
    rfi_max_streak
):
    """
    Update total RFI hit count and max streak for each frequency bin.

    Parameters
    ----------
    rfi_mask_blk : np.ndarray (num_pulses_blk, num_fft)
        Boolean mask indicating RFI detection for each pulse and frequency bin.
    rfi_freq_bin_hit_count : np.ndarray (num_fft,)
        Accumulator for total number of RFI hits per frequency bin.
    rfi_max_streak : np.ndarray (num_fft,)
        Accumulator for maximum consecutive hit streak per frequency bin.
    """

    # Update total hit count
    rfi_freq_bin_hit_count += np.sum(rfi_mask_blk, axis=0)

    # Zero-pad start and end of RFI Mask
    diff = np.diff(np.pad(rfi_mask_blk.astype(int), ((1, 1), (0, 0)), mode='constant'), axis=0)

    # starts and ends are tuples: starts[0]: pulse idx; starts[1]: freq bin idx
    starts = np.where(diff == 1)
    ends = np.where(diff == -1)

    # For each frequency bin, compute max streak
    for b in range(rfi_mask_blk.shape[1]):
        start_idxs = starts[0][starts[1] == b]
        end_idxs = ends[0][ends[1] == b]

        # Expected an end to every beginning
        len(start_idxs) == len(end_idxs)

        streaks = end_idxs - start_idxs
        if len(streaks) > 0:
            rfi_max_streak[b] = max(rfi_max_streak[b], np.max(streaks))

def copy_group_from_input_hdf5(
    input_file, output_file, 
    input_group_path, 
    output_group_path
):
    """
    Copy a group from the input HDf5 file to an output HDF5 file
    """

    with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "a") as f_out:
        if input_group_path in f_in:
            # Check if the group already exists in output and delete it if necessary
            if input_group_path in f_out:
                del f_out[input_group_path]

            # Copy the group
            f_in.copy(input_group_path, f_out[output_group_path])
        else:
            raise KeyError(f"'{input_group_path}' does NOT exist in the input file.")

def process_l0b_data(
    raw: Raw,
    input_file: str,
    output_dir: str,
    num_pulses_blk: int,
    thresh_margin: float = 6.0,
    num_fft: int = 2048,
):
    """
    Process L0B noise-only raw data in chunks to assess RFI power and frequency location
    through frequency domain power spectra density analysis.

    Parameters
    ----------
    raw: Raw
        ISCE3 Raw object
    input_file: str
        Complete file path for input HDF5 file
    output_dir: str
        Output HDF5 file path
    num_pulses_blk: int
        Number of pulses in slow time block
    threshold_margin: float
        After noise power of raw data is estimated, RFI threshold is derived by
        adding this threshold margin to it.
    num_fft: int
        Number of FFT as well number of samples per segment for the Welch Function.
        Default: 2048
    """

    # Output file name is consisted of input file name + SPECTRA.hdf5
    input_filename = os.path.basename(input_file)
    input_base, input_ext = os.path.splitext(input_filename)
    output_file = os.path.join(output_dir, f'{input_base}_SPECTRA{input_ext}')
    
    # Print output file information
    print(f'{output_file = }')
    print()

    with h5py.File(output_file, 'w') as f_out:
        # Write metadata as HDF5 attributes
        f_out.attrs['num_pulses_block'] = num_pulses_blk
        f_out.attrs['num_fft'] = num_fft
        f_out.attrs['psd_x_axis'] = 'range (axis=1)'
        f_out.attrs['psd_y_axis'] = 'power spectra density (axis=0, dB/Hz)'

        for freq, pol_list in raw.polarizations.items(): # A, B
            for pol in pol_list: # HH, HV, VV, VH
                print(f'Start Processing: frequency{freq} {pol}\n')

                # Read raw data
                raw_data = raw.getRawDataset(freq, pol)
                
                pol_tx = pol[0]
                fc, fs, _, _ = raw.getChirpParameters(freq, pol_tx)

                # Estimate noise power of each dataset corresponding to a polarization
                # EVD is used due to unknown RFI situation as a priori
                noise_pwr_pulse_est_db = estimate_noise_pwr_pulse(raw_data, num_pulses_proc) 

                # Convert Time-Domain Pulse Power to Frequency Domain Threshold in dB/Hz
                threshold_db_hz = noise_pwr_pulse_est_db - 10*np.log10(fs) + thresh_margin

                num_pulses, _ = raw_data.shape
                slices = list(slice_gen(num_pulses, num_pulses_blk, combine_rem=False))
                num_az_blocks = len(slices)

                # Compute Frequencies from dummy data: one pulse
                freqs, _ = welch(
                    raw_data[0], 
                    fs=fs, 
                    nperseg=num_fft, 
                    nfft=num_fft, 
                    detrend=False, 
                    return_onesided=False
                )

                # Shift to RF Center Frequency
                freqs = fftshift(freqs) + fc
                num_freq_bins = len(freqs)

                # Create output group for output HDF5
                base_path = f"/science/LSAR/QA/data/frequency{freq}/{pol}"
                out_grp = f_out.require_group(base_path)

                # Center Frequency
                if "centerFrequency" in out_grp:
                    del out_grp["centerFrequency"]
                dset_fc = out_grp.require_dataset('centerFrequency', (), np.float64)
                dset_fc[()] = fc
                dset_fc.attrs["description"] = np.bytes_(
                    f"Raw data center frequency"
                )
                dset_fc.attrs["units"] = np.bytes_("Hz")

                # Sampling Frequency
                if "sampleRate" in out_grp:
                    del out_grp["sampleRate"]
                dset_fs = out_grp.create_dataset('sampleRate', (), np.float64)
                dset_fs[()] = fs
                dset_fs.attrs["description"] = np.bytes_(
                    f"Raw data sampling frequency"
                )
                dset_fs.attrs["units"] = np.bytes_("Hz")

                # Power Spectra Density
                if "rangePowerSpectralDensity" in out_grp:
                    del out_grp["rangePowerSpectralDensity"]

                # Create new dataset
                dset_psd = out_grp.create_dataset(
                    'rangePowerSpectralDensity',
                    shape=(num_az_blocks, num_freq_bins),
                    dtype=np.float32,
                    compression='gzip'
                )

                # Write noise power estimate per pulse
                if "noisePowerEstimate" in out_grp:
                    del out_grp["Frequency"]
                dset_noise = out_grp.create_dataset("noisePowerEstimate", data=noise_pwr_pulse_est_db)
                dset_noise.attrs["description"] =  np.bytes_(
                    f"Estimate of noise floor"
                )
                dset_noise.attrs["units"] = np.bytes_("decibel re 1 DN^2")

                # Write frequency domain RFI threshold estimate
                if "rfiDetectionThreshold" in out_grp:
                    del out_grp["rfiDetectionThreshold"]
                dset_thresh = out_grp.create_dataset('rfiDetectionThreshold', data=threshold_db_hz)
                dset_thresh.attrs["description"] =  np.bytes_(
                    f"Threshold used to classify power spectral density values as contaminated "
                    f"by radio frequency interference"
                )
                dset_thresh.attrs["units"] = np.bytes_("decibel re 1/hertz")

                # Initialize hit count array
                rfi_bin_hit_count = np.zeros(num_freq_bins, dtype=int)
                rfi_bin_max_streak = np.zeros(num_freq_bins, dtype=int) 
                
                # Compute averaged power spectra density and RFI hit count
                for az_idx, az_slice in enumerate(slices):
                    az_blk = raw_data[az_slice, :]  # Shape: (num_pulses_blk, range_samples)
                    _, psd_az_blk = welch(
                        az_blk, 
                        fs=fs, 
                        nperseg=num_fft, 
                        nfft=num_fft, 
                        detrend=False, 
                        axis=1, 
                        return_onesided=False
                    )
                    psd_az_blk_db = 10*np.log10(fftshift(psd_az_blk))

                    # Compare each frequency bin in each pulse against the threshold
                    rfi_mask_blk = psd_az_blk_db > threshold_db_hz  # shape (num_pulses_blk, num_fft)

                    # Update RFI stats: Collect RFI hit data, and RFI hit max streak data
                    update_rfi_hit_stats(rfi_mask_blk, rfi_bin_hit_count, rfi_bin_max_streak)

                    # Compute Average range frequency power spectra density (dB/Hz)
                    avg_psd_az_blk = np.mean(fftshift(psd_az_blk), axis=0)
                    avg_psd_az_blk_db = 10 * np.log10(avg_psd_az_blk)

                    # Write PSD of each block
                    dset_psd[az_idx, :] = avg_psd_az_blk_db

                    if (az_idx + 1) % 10 == 0 or (az_idx + 1) == num_az_blocks:
                        print(f"Processed {az_idx + 1}/{num_az_blocks} blocks ({(az_idx + 1)/num_az_blocks*100:.1f}%)")

                # Write RFI Hit Count
                if "interferenceHitCount" in out_grp:
                    del out_grp["interferenceHitCount"]
                dset_hit = out_grp.create_dataset('interferenceHitCount', data=rfi_bin_hit_count)
                dset_hit.attrs["description"] =  np.bytes_(
                    f"Count of pulses where RFI detection threshold was exceeded "
                    f"(in each frequency bin)"
                )
                dset_hit.attrs["units"] = np.bytes_("1")

                # Write RFI Hit Max Streak
                if "interferenceMaxStreak" in out_grp:
                    del out_grp["interferenceMaxStreak"]
                dset_streak = out_grp.create_dataset('interferenceMaxStreak', data=rfi_bin_max_streak)
                dset_streak.attrs["description"] =  np.bytes_(
                    f"Maximum number of consecutive pulses where RFI detection threshold was exceeded "
                    f"(in each freuqency bin)"
                )
                dset_streak.attrs["units"] = np.bytes_("1")

                print(f'Processed: frequency{freq} {pol}')
                print()
            
            dset_psd.attrs["description"] =  np.bytes_(
                "Radio frequency coordinates for range power spectra (dtype: dataset[list[float64]])"
            )
            dset_psd.attrs["units"] = np.bytes_("decibel re 1/hertz")

            # Frequency
            top_level_path = f"/science/LSAR/QA/data/frequency{freq}"
            top_grp = f_out.require_group(top_level_path)

            # Polarizations
            if "listOfPolarizations" in top_grp:
                del top_grp["listOfPolarizations"]
            dset_pols = top_grp.create_dataset('listOfPolarizations', data=pol_list)
            dset_pols.attrs["description"] =  np.bytes_(
                f"Polarizations for Frequency A discovered in input NISAR product"
            )

            # Range Spectra Frequencies
            if "rangeSpectraFrequencies" in top_grp:
                del top_grp["rangeSpectraFrequencies"]
            dset_freq = top_grp.create_dataset('rangeSpectraFrequencies', data=freqs)
            dset_freq.attrs["description"] =  np.bytes_(
                f"Radio Frequency (RF) frequency coordinates for range power spectra"
            )
            dset_freq.attrs["units"] = np.bytes_("decibel re 1/hertz")

            # Threshold margin in db/Hz
            if "thresholdMargin" in top_grp:
                del top_grp["thresholdMargin"]
            dset_margin = top_grp.create_dataset('thresholdMargin', data=thresh_margin)
            dset_margin.attrs["description"] =  np.bytes_(
                f"RFI frequency domain detection margin"
            )
            dset_margin.attrs["units"] = np.bytes_("decibel")

        # Copy Identification group from input L0B file
        input_group_path = os.path.join(raw._RootPath, raw._IdentificationPath)
        output_group_path = raw._RootPath

        copy_group_from_input_hdf5(input_file, output_file, input_group_path, output_group_path)

if __name__ == "__main__":
    inputs = cmd_line_parse()

    input_file = inputs.input_file
    output_dir = inputs.output_dir
    num_pulses_proc = inputs.num_pulses_proc
    num_pulses_blk = inputs.num_pulses_blk
    num_fft = inputs.num_fft
    thresh_margin = inputs.thresh_margin
    
    raw = Raw(hdf5file=input_file)
    raw.parsePolarizations()

    process_l0b_data(
        raw,
        input_file,
        output_dir,
        num_pulses_blk,
        thresh_margin,
        num_fft,
    )
