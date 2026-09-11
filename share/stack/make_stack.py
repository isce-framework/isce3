#!/usr/bin/env python3

import argparse

from StackProcessor import StackProcessor


STEPS = [
    "coarse",
    "dense",
    "invert",
    "rubbersheet",
    "resample",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NISAR StackProcessor pipeline."
    )

    parser.add_argument(
        "--slc-folder",
        required=True,
        help="Folder containing NISAR RSLC HDF5 files.",
    )

    parser.add_argument(
        "--out-folder",
        required=True,
        help="Output stack folder.",
    )

    parser.add_argument(
        "--dem-file",
        required=True,
        help="DEM file.",
    )

    parser.add_argument(
        "--pair-level",
        type=int,
        default=2,
        help="Dense-offset pair level (default: 2).",
    )

    parser.add_argument(
        "--freq",
        default="A",
        help="Frequency, e.g. A (default: A).",
    )

    parser.add_argument(
        "--pol",
        default="HH",
        help="Polarization (default: HH).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable StackProcessor debug mode.",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=STEPS,
        help=(
            "Pipeline stages to run. "
            "Default: coarse dense invert rubbersheet resample"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    processor = StackProcessor(
        slc_folder=args.slc_folder,
        out_folder=args.out_folder,
        dem_file=args.dem_file,
        pair_level=args.pair_level,
        freq=args.freq,
        pol=args.pol,
        overwrite=args.overwrite,
        debug=args.debug,
    )

    if "coarse" in args.steps:
        print("\n===== COARSE REGISTER =====")
        processor.coarse_register_scans()

    if "dense" in args.steps:
        print("\n===== DENSE OFFSETS =====")
        processor.dense_offset_pairs()

    if "invert" in args.steps:
        print("\n===== INVERT DENSE OFFSETS =====")
        processor.invert_dense_offset_pairs()

    if "rubbersheet" in args.steps:
        print("\n===== RUBBERSHEET =====")
        processor.rubbersheet()

    if "resample" in args.steps:
        print("\n===== RESAMPLE SLCs =====")
        processor.resample_slcs()

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()
