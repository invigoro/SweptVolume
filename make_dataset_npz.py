#!/usr/bin/env python3
"""Pack the six .npy arrays into the swept_volume_data.npz file the assignment expects.

The graded submission is run with swept_volume_data.npz placed in the extracted
submission directory. During local development the arrays live unpacked in
source_data/swept_volume_data/, so this script builds the expected file:

    python3 make_dataset_npz.py

This is a development convenience only. The grader supplies the .npz directly,
and neither source_data/ nor swept_volume_data.npz is included in the submission.
"""

import argparse
import pathlib
import sys

import numpy as np

ARRAY_NAMES = [
    "training_features",
    "training_labels",
    "evaluation_features",
    "evaluation_labels",
    "testing_features",
    "testing_labels",
]

DEFAULT_SOURCE = pathlib.Path("source_data/swept_volume_data")
DEFAULT_OUTPUT = pathlib.Path("swept_volume_data.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE,
                    help=f"directory holding the .npy arrays (default: {DEFAULT_SOURCE})")
    ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                    help=f"output .npz path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite the output file if it already exists")
    args = ap.parse_args()

    if args.output.exists() and not args.replace:
        print(f"{args.output} already exists; pass --replace to overwrite.", file=sys.stderr)
        return 1

    arrays = {}
    for name in ARRAY_NAMES:
        path = args.source / f"{name}.npy"
        if not path.exists():
            print(f"missing required array: {path}", file=sys.stderr)
            return 1
        arrays[name] = np.load(path)

    for name, value in arrays.items():
        print(f"  {name}: {value.shape} shape, type {value.dtype}.")

    np.savez_compressed(args.output, **arrays)
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nwrote {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
