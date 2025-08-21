#!/usr/bin/env python3
"""
Quick viewer for .npz archives. Prints the first few values of each array.

Usage examples:
  python scripts/peek_npz.py                          # uses default path
  python scripts/peek_npz.py /path/to/file.npz       # custom file
  python scripts/peek_npz.py --list-keys             # just list keys
  python scripts/peek_npz.py -k 5 -e 20              # limit keys/elements
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a quick preview of arrays stored in a .npz file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="/data/rohith/ag/videos/results/result.npz",
        help="Path to the .npz file",
    )
    parser.add_argument(
        "-k",
        "--max-keys",
        type=int,
        default=10,
        help="Maximum number of arrays (keys) to preview",
    )
    parser.add_argument(
        "-e",
        "--max-elements",
        type=int,
        default=16,
        help="Maximum number of elements to print per array (flattened)",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Only list array keys in the archive and exit",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading pickled objects (use with caution)",
    )
    return parser.parse_args(list(argv))


def format_array_preview(array: np.ndarray, max_elements: int) -> str:
    """Return a concise string showing the first few elements of the array.

    We flatten to keep output compact and consistent across shapes.
    """
    flattened = array.ravel()
    preview = flattened[: max(0, max_elements)]
    return np.array2string(
        preview,
        threshold=max(0, max_elements),
        edgeitems=max(0, max_elements),
        max_line_width=120,
    )


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.path):
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        return 1

    try:
        archive = np.load(args.path, allow_pickle=bool(args.allow_pickle))
    except Exception as exc:  # noqa: BLE001
        print(f"Error: failed to load {args.path}: {exc}", file=sys.stderr)
        return 1

    keys = list(archive.keys())
    if not keys:
        print("Archive contains no arrays.")
        return 0

    if args.list_keys:
        print(f"Found {len(keys)} arrays in {args.path}:")
        for key in keys:
            try:
                arr = archive[key]
                print(f"- {key}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception as exc:  # noqa: BLE001
                print(f"- {key}: <error reading array: {exc}>")
        return 0

    print(f"Previewing up to {min(args.max_keys, len(keys))} of {len(keys)} arrays in {args.path}\n")

    shown = 0
    for key in keys:
        if shown >= args.max_keys:
            remaining = len(keys) - shown
            if remaining > 0:
                print(f"… ({remaining} more arrays not shown)")
            break
        try:
            arr = archive[key]
            print(f"[{shown+1}] key='{key}' | shape={arr.shape} | dtype={arr.dtype}")
            print(format_array_preview(arr, args.max_elements))
            print()  # blank line between arrays
            shown += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{shown+1}] key='{key}' | <error reading array: {exc}>")
            print()
            shown += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
