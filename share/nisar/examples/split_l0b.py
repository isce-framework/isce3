#!/usr/bin/env python3
import argparse
import h5py
import nisar
import numpy as np
import re


log = print


def getargs():
    parser = argparse.ArgumentParser(description="Split a raw data file "
        "into two parts in order to test mixed-mode processing and "
        "file recording gap scenarios.")
    parser.add_argument("--l0b", metavar="FILENAME", required=True,
        help="Raw data file in NISAR L0B format.")
    parser.add_argument("--out1", metavar="FILENAME", required=True)
    parser.add_argument("--out2", metavar="FILENAME", required=True)
    parser.add_argument("--gap", type=float, default=0.0,
        help="Duration of gap in seconds to leave between output files.")
    parser.add_argument("--location", type=float, default=None,
        help=("Location to split file, specified as azimuth time in seconds "
            "since first pulse (default=middle)"))
    return parser.parse_args()


def truncate_azimuth_dataset(dset, fout, az_slice):
    # Assume slow axis is azimuth time.  True except for DM2.
    log(f"shape before = {dset.shape}")
    x = dset[az_slice]
    log(f"shape after = {x.shape}")
    dset_out = fout.require_dataset(dset.name, data=x, dtype=dset.dtype,
        shape=x.shape, chunks=dset.chunks, compression=dset.compression,
        compression_opts=dset.compression_opts)
    for key in dset.attrs:
        dset_out.attrs[key] = dset.attrs[key]


patterns_to_truncate = [re.compile(s) for s in (
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/radarTime",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/rangeLineIndex",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/UTCtime",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/validSamplesSubSwath\d",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/txPhase",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/chirpCorrelator",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/calType",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/attenuation",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/caltone",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/rx[HV]/[HVLR][HV]",
    r"science/.SAR/RRSD/swaths/frequency[AB]/tx[HVLR]/rx[HV]/[RD|WD|WL]",
)]


def copy_or_truncate(h5in, h5out, az_slice, path):
    obj = h5in[path]
    if any(pattern.match(path) for pattern in patterns_to_truncate):
        log(f"truncate: {path}")
        truncate_azimuth_dataset(obj, h5out, az_slice)
    elif isinstance(obj, h5py.Group):
        log(f"group: {path}")
        g = h5out.require_group(path)
        g.attrs.update(obj.attrs)
    elif isinstance(obj, h5py.Dataset):
        log(f"dataset: {path}")
        h5out.copy(obj, path)
    else:
        log(f"skipping: {path}")


def main(args):
    # Get timing data.
    reader = nisar.products.readers.Raw.open_rrsd(args.l0b)
    freq = reader.frequencies[0]
    txpol = reader.polarizations[freq][0][0]
    epoch, times = reader.getPulseTimes(freq, txpol)
    prf = 1.0 / np.mean(np.diff(times))

    # Figure out where to split the data.
    times -= times[0]
    location = args.location
    if location is None:
        location = times[len(times) // 2]
    i0 = np.argmin(np.abs(times - (location - args.gap / 2)))
    i1 = np.argmin(np.abs(times - (location + args.gap / 2)))
    log(f"First chunk covers pulses [0, {i0}]")
    log(f"Second chunk covers pulses [{i1}, {len(times)}]")

    # Open input file.
    with h5py.File(args.l0b, mode="r") as src_h5:
    
        for name, mask in ((args.out1, slice(0, i0)), (args.out2, slice(i1, None))):
            # Create output file.
            with h5py.File(name, mode="w") as dest_h5:
                # Copy everything, truncating the datasets with an azimuth dimension.
                src_h5.visit(lambda path: copy_or_truncate(src_h5, dest_h5, mask, path))


if __name__ == "__main__":
    main(getargs())
